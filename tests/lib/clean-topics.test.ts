import { describe, expect, it } from "vitest";
import { cleanTopic, cleanTopics, fillActionsFromRequestText, repairActions } from "../../src/lib/clean-topics.js";
import type { Topic } from "../../src/lib/schema.js";

const base: Topic = {
  domain: "roads",
  issue: "яма на дорозі",
  object: "вул. Перемоги, 5",
  requested_action: "відремонтувати",
  attributes: {},
};

describe("cleanTopic", () => {
  it("strips trailing applicant narration from issue", () => {
    const out = cleanTopic({
      ...base,
      issue: "третій тиждень не працює вуличне освітлення — темно, діти йдуть додому в темряві. заявниці.",
      requested_action: "",
    });
    expect(out.issue).toBe(
      "третій тиждень не працює вуличне освітлення — темно, діти йдуть додому в темряві.",
    );
  });

  it("strips trailing applicant narration from requested_action", () => {
    const out = cleanTopic({
      ...base,
      requested_action: "відремонтувати тротуар. заявник.",
    });
    expect(out.requested_action).toBe("відремонтувати тротуар.");
  });

  it("moves an explicit request sentence from issue into empty requested_action without duplicating Прошу", () => {
    const out = cleanTopic({
      ...base,
      issue: "на тротуарі біля зупинки утворилася яма. Прошу відремонтувати тротуар.",
      requested_action: "",
    });
    expect(out.issue).toBe("на тротуарі біля зупинки утворилася яма.");
    expect(out.requested_action).toBe("відремонтувати тротуар.");
  });

  it("removes a duplicate request sentence from issue but keeps existing requested_action", () => {
    const out = cleanTopic({
      ...base,
      issue: "вулиця Темна. Прошу встановити ліхтарі.",
      requested_action: "встановити ліхтарі на вулиці Темній",
    });
    expect(out.issue).toBe("вулиця Темна.");
    expect(out.requested_action).toBe("встановити ліхтарі на вулиці Темній");
  });

  it("leaves a clean topic unchanged", () => {
    expect(cleanTopic(base)).toEqual(base);
  });

  it("preserves domain, object and attributes", () => {
    const topic: Topic = {
      domain: "electricity",
      issue: "не горять ліхтарі",
      object: "вул. Шевченка, 24",
      requested_action: "відновити освітлення",
      attributes: { street: "Шевченка" },
    };
    expect(cleanTopic(topic)).toEqual(topic);
  });
});

describe("cleanTopics", () => {
  it("cleans every topic and keeps the topics wrapper", () => {
    const out = cleanTopics({
      topics: [
        { ...base, issue: "темно. заявниці.", requested_action: "" },
        { ...base, issue: "чиста проблема" },
      ],
    });
    expect(out.topics).toHaveLength(2);
    expect(out.topics[0]!.issue).toBe("темно.");
    expect(out.topics[1]!.issue).toBe("чиста проблема");
  });

  it("returns an empty topics array unchanged", () => {
    expect(cleanTopics({ topics: [] })).toEqual({ topics: [] });
  });
});

describe("fillActionsFromRequestText", () => {
  const empty = (domain: Topic["domain"]): Topic => ({ ...base, domain, requested_action: "" });

  it("splits a compound request across all-empty topics in order", () => {
    const out = fillActionsFromRequestText(
      { topics: [empty("electricity"), empty("roads")] },
      "Прошу відновити освітлення та відремонтувати тротуар.",
    );
    expect(out.topics[0]!.requested_action).toBe("відновити освітлення");
    expect(out.topics[1]!.requested_action).toBe("відремонтувати тротуар");
  });

  it("keeps a compound request whole when there is only one empty topic", () => {
    const out = fillActionsFromRequestText(
      { topics: [empty("roads")] },
      "Прошу відремонтувати тротуар та відновити розмітку.",
    );
    expect(out.topics[0]!.requested_action).toBe("відремонтувати тротуар та відновити розмітку");
  });

  it("assigns multiple request sentences sequentially", () => {
    const out = fillActionsFromRequestText(
      { topics: [empty("electricity"), empty("roads")] },
      "Прошу відновити освітлення. Просимо відремонтувати тротуар.",
    );
    expect(out.topics[0]!.requested_action).toBe("відновити освітлення");
    expect(out.topics[1]!.requested_action).toBe("відремонтувати тротуар");
  });

  it("returns unchanged when the source has no request sentence", () => {
    const input = { topics: [empty("roads")] };
    const out = fillActionsFromRequestText(input, "На вулиці темно вже тиждень.");
    expect(out.topics[0]!.requested_action).toBe("");
  });

  it("fills an empty topic with a request part no other topic claimed", () => {
    const input = { topics: [{ ...base, domain: "roads" as const, requested_action: "заповнено" }, empty("water")] };
    const out = fillActionsFromRequestText(input, "Прошу відновити освітлення.");
    expect(out.topics[0]!.requested_action).toBe("заповнено");
    expect(out.topics[1]!.requested_action).toBe("відновити освітлення");
  });

  it("leaves an empty topic alone when every request part is already claimed", () => {
    const input = {
      topics: [
        { ...base, domain: "heating" as const, requested_action: "" },
        { ...base, domain: "housing" as const, requested_action: "відновити подачу тепла та відремонтувати ліфт" },
      ],
    };
    const out = fillActionsFromRequestText(input, "Прошу відновити подачу тепла та відремонтувати ліфт.");
    expect(out.topics[0]!.requested_action).toBe("");
    expect(out.topics[1]!.requested_action).toBe("відновити подачу тепла та відремонтувати ліфт");
  });

  it("ignores sentences that merely mention the user inside the request", () => {
    const out = fillActionsFromRequestText(
      { topics: [empty("sanitation")] },
      "Вимагаю прибрати сміття!",
    );
    expect(out.topics[0]!.requested_action).toBe("прибрати сміття");
  });
});

describe("repairActions", () => {
  const empty = (domain: Topic["domain"]): Topic => ({ ...base, domain, requested_action: "" });

  it("extends a truncated action that is a prefix of the user's request", () => {
    const out = repairActions(
      {
        topics: [
          { ...empty("heating"), issue: "немає опалення, в квартирах холодно" },
          {
            ...empty("housing"),
            issue: "зламаний ліфт. відновити подачу тепла та відремонтувати ліфт",
            requested_action: "відновити подачу тепла та відремен",
          },
        ],
      },
      "Прошу відновити подачу тепла та відремонтувати ліфт.",
    );
    expect(out.topics[1]!.requested_action).toBe("відновити подачу тепла та відремонтувати ліфт");
  });

  it("removes an issue sentence that duplicates the topic's action", () => {
    const out = repairActions(
      {
        topics: [
          {
            ...empty("housing"),
            issue: "зламаний ліфт. відремонтувати ліфт",
            requested_action: "відремонтувати ліфт",
          },
        ],
      },
      "Прошу відремонтувати ліфт.",
    );
    expect(out.topics[0]!.issue).toBe("зламаний ліфт.");
    expect(out.topics[0]!.requested_action).toBe("відремонтувати ліфт");
  });

  it("keeps an exact action unchanged", () => {
    const out = repairActions(
      { topics: [{ ...empty("roads"), issue: "яма на дорозі", requested_action: "відремонтувати тротуар" }] },
      "Прошу відремонтувати тротуар.",
    );
    expect(out.topics[0]!.requested_action).toBe("відремонтувати тротуар");
    expect(out.topics[0]!.issue).toBe("яма на дорозі");
  });

  it("leaves an unrelated action untouched", () => {
    const out = repairActions(
      { topics: [{ ...empty("roads"), requested_action: "прибрати сміття" }] },
      "Прошу відремонтувати тротуар.",
    );
    expect(out.topics[0]!.requested_action).toBe("прибрати сміття");
  });

  it("does not empty an issue whose only sentence duplicates the action", () => {
    const out = repairActions(
      { topics: [{ ...empty("housing"), issue: "відремонтувати ліфт", requested_action: "відремонтувати ліфт" }] },
      "Прошу відремонтувати ліфт.",
    );
    expect(out.topics[0]!.issue).toBe("відремонтувати ліфт");
  });

  it("returns unchanged when the source has no request sentence", () => {
    const input = { topics: [{ ...empty("roads"), requested_action: "щось" }] };
    expect(repairActions(input, "На вулиці темно.")).toEqual(input);
  });
});
