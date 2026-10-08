import { StructuredComplaint } from "../lib/schema.js";
import type { ComplaintInput } from "../lib/schema.js";
import { normalizeComplaintInput } from "../lib/normalize.js";
import { cleanTopics, fillActionsFromRequestText, repairActions } from "../lib/clean-topics.js";

export interface InferenceResult {
  structured: StructuredComplaint;
  raw: string;
}

export class InferenceError extends Error {
  constructor(message: string, public readonly raw?: string) {
    super(message);
    this.name = "InferenceError";
  }
}

const SYSTEM_PROMPT = `Ти — система, що перетворює звернення громадян до органів місцевого самоврядування у структурований список проблем. Кожна проблема має точно 5 полів: domain, issue, object, requested_action, attributes.
Вимоги:
- Відповідай ТІЛЬКИ валідним JSON-масивом. Не додавай текст до/після.
- Без роздумів, без markdown, без \`!\`, без \`>\` або інших префіксів.
- У кожній проблемі поле \`domain\` ОБОВ’ЯЗКОВО має бути одним із: roads, water, heating, housing, transport, sanitation, electricity, construction, benefits, government, commerce, payments, other.
- \`issue\` — стислий опис проблеми (українською): ТІЛЬКИ стан/факт. Не вживай слів «прошу», «заявник», «заявниця», «громадянин» і не переказуй прохання — прохання належить полю \`requested_action\`.
- \`object\` — об'єкт звернення як у тексті користувача, з номером будинку і в називному відмінку (напр. «вул. Грушевського, 12»). Якщо не вказано — використай "не вказано".
- \`requested_action\` — ОБОВ’ЯЗКОВО заповнюй, коли у тексті є хоч одне прохання, вимога чи дія (прошу, треба, потрібно, вимагаю, відновити, відремонтувати, прибрати тощо): винеси його туди одним реченням, без слова «прошу» на початку. Порожній рядок (\`\`) — лише якщо прохання справді немає. Не дублюй прохання в \`issue\`.
- \`attributes\` — об'єкт з додатковими деталями (значення — рядки). Якщо деталей немає — порожній об'єкт {}.
- Розділяй декілька проблем окремими об'єктами в масиві. Не об'єднуй їх.
- Після крапки не додавай слів-описів («заявниці», «заявник», «громадянин») та жодних зайвих хвостів: кожне поле містить лише свій зміст. Не включай персональних даних (ПІБ, номери телефонів).

Приклад 1.
Вхід: «На вулиці Соборній уже тиждень не горять ліхтарі. Прошу відновити освітлення.»
Відповідь: [{"domain":"electricity","issue":"На вулиці Соборній уже тиждень не горять ліхтарі","object":"не вказано","requested_action":"відновити освітлення","attributes":{}}]

Приклад 2.
Вхід: «Біля будинку 5 на провулку Лісовому зламана труба, заливає двір. Треба терміново усунути аварію. Крім того, не вивозять сміття вже другий тиждень.»
Відповідь: [{"domain":"water","issue":"Біля будинку 5 на провулку Лісовому зламана труба, заливає двір","object":"пров. Лісовий, 5","requested_action":"терміново усунути аварію","attributes":{}},{"domain":"sanitation","issue":"Не вивозять сміття вже другий тиждень","object":"не вказано","requested_action":"вивезти сміття","attributes":{}}]`;

function extractJsonValueAt(s: string, from: number): { value: string; end: number } | null {
  const rel = s.slice(from).search(/[{[]/);
  if (rel < 0) return null;
  const start = from + rel;
  const stack: string[] = [];
  const closing: Record<string, string> = { "{": "}", "[": "]" };
  let inString = false;
  let escaped = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i]!;
    if (inString) {
      if (escaped) escaped = false;
      else if (c === "\\") escaped = true;
      else if (c === '"') inString = false;
      continue;
    }
    if (c === '"') {
      inString = true;
      continue;
    }
    if (c === "{" || c === "[") {
      stack.push(c);
    } else if (c === "}" || c === "]") {
      if (stack.length === 0) return null;
      const top = stack.pop()!;
      if (closing[top] !== c) return null;
      if (stack.length === 0) return { value: s.slice(start, i + 1), end: i + 1 };
    }
  }
  return null;
}

function tryParseAt(s: string, start: number): StructuredComplaint | null {
  const values: string[] = [];
  let pos = start;
  for (;;) {
    const hit = extractJsonValueAt(s, pos);
    if (!hit) break;
    values.push(hit.value);
    let j = hit.end;
    while (j < s.length && /\s/.test(s[j]!)) j++;
    if (s[j] !== ",") break;
    pos = j + 1;
  }
  if (values.length === 0) return null;
  const jsonStr = values.length === 1 ? values[0]! : `[${values.join(",")}]`;
  try {
    const parsed = JSON.parse(jsonStr);
    if (Array.isArray(parsed)) return StructuredComplaint.parse({ topics: parsed });
    if (parsed && typeof parsed === "object") {
      const obj = parsed as Record<string, unknown>;
      if (Array.isArray(obj.topics)) return StructuredComplaint.parse(obj);
      if ("domain" in obj) return StructuredComplaint.parse({ topics: [obj] });
    }
    return StructuredComplaint.parse(parsed);
  } catch {
    return null;
  }
}

interface CachedIdentityToken {
  value: string;
  expiresAt: number;
}

let cachedIdentityToken: CachedIdentityToken | null = null;

const IDENTITY_TOKEN_URL =
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity";
const IDENTITY_TOKEN_TTL_MS = 50 * 60 * 1000;
const IDENTITY_TOKEN_REFRESH_MARGIN_MS = 30 * 1000;

export async function fetchIdentityToken(audience: string): Promise<string> {
  if (cachedIdentityToken && cachedIdentityToken.expiresAt > Date.now() + IDENTITY_TOKEN_REFRESH_MARGIN_MS) {
    return cachedIdentityToken.value;
  }
  let resp: Response;
  try {
    resp = await fetch(`${IDENTITY_TOKEN_URL}?audience=${encodeURIComponent(audience)}`, {
      headers: { "Metadata-Flavor": "Google" },
      signal: AbortSignal.timeout(3000),
    });
  } catch {
    throw new InferenceError("Identity token fetch failed");
  }
  if (!resp.ok) {
    throw new InferenceError(`Identity token fetch failed with status ${resp.status}`);
  }
  const value = (await resp.text()).trim();
  if (!value) {
    throw new InferenceError("Identity token fetch returned an empty token");
  }
  cachedIdentityToken = { value, expiresAt: Date.now() + IDENTITY_TOKEN_TTL_MS };
  return value;
}

export function clearIdentityTokenCache(): void {
  cachedIdentityToken = null;
}

export function parseModelContent(content: string): StructuredComplaint {
  const str = String(content);
  for (let i = 0; i < str.length; i++) {
    const c = str[i];
    if (c !== "{" && c !== "[") continue;
    const out = tryParseAt(str, i);
    if (out) return out;
  }
  throw new InferenceError("No parseable JSON topic structure in model output");
}

const REQUEST_MARKER = /(?<!\p{L})(?:прошу|просимо|треба|потрібно|вимага|прохання|звертаю|наголошую)/iu;
const RETRY_REMINDER =
  "\nНАГАДУВАННЯ: користувач явно просить виконати дію — requested_action має бути заповнений для кожної проблеми, до якої стосується прохання.";

function hasRequestMarker(text: string): boolean {
  return REQUEST_MARKER.test(text);
}

export async function analyzeComplaint(input: ComplaintInput): Promise<InferenceResult> {
  const normalized = normalizeComplaintInput(input);
  const inferenceUrl = process.env.INFERENCE_SERVICE_URL;

  if (!inferenceUrl) {
    // Fallback to local if not configured (dev)
    const { analyzeComplaint: localAnalyze } = await import("./inference_local.js");
    return localAnalyze(input);
  }

  const origin = inferenceUrl.replace(/\/$/, "");
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (process.env.INFERENCE_AUTH === "metadata") {
    const token = await fetchIdentityToken(origin);
    headers["Authorization"] = `Bearer ${token}`;
  }

  const callModel = async (systemPrompt: string): Promise<InferenceResult> => {
    const body = {
      model: "qwen3-8b-v2",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: normalized.text },
      ],
      temperature: 0,
      max_tokens: 800,
      stream: false,
      stop: ["\n!\n"],
    };
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      const resp = await fetch(origin + "/v1/chat/completions", {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new InferenceError(`Inference service error ${resp.status}: ${text}`);
      }
      const data = await resp.json();
      const content = data?.choices?.[0]?.message?.content;
      if (!content) {
        throw new InferenceError("No content in inference response");
      }
      const str = String(content);
      const structured = repairActions(cleanTopics(parseModelContent(str)), normalized.text);
      return { structured, raw: str };
    } catch (err) {
      if (err instanceof InferenceError) throw err;
      throw new InferenceError(err instanceof Error ? err.message : String(err));
    } finally {
      clearTimeout(timeout);
    }
  };

  let result = await callModel(SYSTEM_PROMPT);
  const fillAttempt = (r: InferenceResult): InferenceResult => {
    const filled = fillActionsFromRequestText(r.structured, normalized.text);
    const changed = filled.topics.some(
      (t, i) => t.requested_action !== r.structured.topics[i]?.requested_action,
    );
    return changed ? { ...r, structured: filled } : r;
  };
  const filledCount = (r: InferenceResult): number =>
    r.structured.topics.filter((t) => t.requested_action.trim()).length;

  if (hasRequestMarker(normalized.text)) {
    result = fillAttempt(result);
    const gaps = result.structured.topics.length - filledCount(result);
    if (gaps > 0) {
      try {
        const retry = await callModel(SYSTEM_PROMPT + RETRY_REMINDER);
        if (filledCount(retry) > filledCount(result)) {
          result = retry;
        }
        result = fillAttempt(result);
        console.error(
          `[inference-debug] retry ran filled=${filledCount(result)}/${result.structured.topics.length}`,
        );
      } catch (err) {
        console.error(
          `[inference-debug] retry failed: ${err instanceof Error ? err.message : String(err)}`,
        );
        // keep the first successful result; an empty requested_action is editable in review
      }
    } else {
      console.error(
        `[inference-debug] marker path filled=${filledCount(result)}/${result.structured.topics.length} no retry`,
      );
    }
  } else {
    console.error(`[inference-debug] no request marker skip fill/retry`);
  }
  return result;
}
