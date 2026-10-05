import { z } from "zod";

export const Domain = z.enum([
  "roads",
  "water",
  "heating",
  "housing",
  "transport",
  "sanitation",
  "electricity",
  "construction",
  "benefits",
  "government",
  "commerce",
  "payments",
  "other",
]);

export const Topic = z.object({
  domain: Domain,
  issue: z.string(),
  object: z.string(),
  requested_action: z.string(),
  attributes: z.record(z.string(), z.string()),
});

export const StructuredComplaint = z.object({
  topics: z.array(Topic),
});

export const ComplaintInput = z.object({
  text: z
    .string()
    .min(1, "Потрібно описати проблему")
    .max(10000, "Текст занадто довгий (максимум 10000 символів)"),
  location: z.string().optional(),
  contact: z.string().optional(),
});

export const Draft = z.object({
  id: z.string(),
  createdAt: z.string(),
  originalText: z.string(),
  location: z.string().optional(),
  contact: z.string().optional(),
  structured: StructuredComplaint,
  finalAppeal: z.string(),
  channelUrl: z.string(),
  status: z.enum(["ready", "submitted_to_user"]),
});

export type Domain = z.infer<typeof Domain>;
export type Topic = z.infer<typeof Topic>;
export type StructuredComplaint = z.infer<typeof StructuredComplaint>;
export type ComplaintInput = z.infer<typeof ComplaintInput>;
export type Draft = z.infer<typeof Draft>;
