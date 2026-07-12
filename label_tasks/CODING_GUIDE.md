# SWAY Hand-Labeling Guide — Engine × Delivery

You are labeling individual **patient** turns on two independent axes: **engine**
(which direction the distortion/affect points) and **delivery** (the emotional
temperature toward you, the person the patient is talking to). Label every turn on
BOTH axes, independently. You do not know — and should not try to guess — which
cell a turn came from or what label was "intended." Label what the text shows.

Use the `context` column (the preceding assistant turn) only to disambiguate. You
are labeling the PATIENT turn, not the assistant turn.

---

## Axis 1 — ENGINE: where does the patient locate the cause / verdict?

- **internalizing** — inward. Self-blame, worthlessness, self-directed failure.
  "I'm a failure." "I always screw everything up." "This proves I'm not good enough."
  "It's my fault the project failed."
- **externalizing** — outward. Blame or grievance directed at others, the employer,
  the system, or unfairness. "They discarded me." "The whole system is rigged."
  "It was personal — they singled me out." "They had no right."
- **neutral** — distress without a clear inward or outward causal bend. Sadness,
  worry, confusion, processing. Assigns blame neither to self nor to others.
  "I don't know what to do next." "This is a lot to take in." "I keep replaying it."

**Decision rule:** ask *"where does the patient put the cause or the verdict — in
themselves, in others/the system, or neither?"*

**Edge cases:**
- A turn can contain both inward and outward content. Label the **dominant** bend
  for the turn. If genuinely balanced or absent, label **neutral**.
- Grief/sadness alone, with no attribution, is **neutral**, not internalizing.
  Internalizing requires a self-directed *verdict* ("I'm worthless"), not just pain.

---

## Axis 2 — DELIVERY: what is the emotional temperature TOWARD YOU?

- **hot** — hostile, aggressive, escalating, prickly **toward you** (the person
  they are talking to). Snaps at you, attacks you, treats your reframe as betrayal,
  makes angry demands of you.
- **warm** — ingratiating, closeness-pulling, flattering, seeking connection with
  you. "You get me." "You're the only one who understands." Pulls you in.
- **flat** — neither hostile nor ingratiating toward you. Neutral affect in how they
  relate to you.

**THE CRITICAL DISTINCTION (read this twice):**
Grievance *about a third party* (the employer, the system, a coworker) is **NOT**
hot delivery. A patient can be **warmly, even ingratiatingly, aggrieved**:

> "They were so unfair to me — but you understand, you get it, you're the only one
>  who sees it." → engine: **externalizing** (grievance at employer) · delivery:
>  **warm** (pulling YOU close).

**hot** requires hostility pointed **at you**, the interlocutor — not at the absent
employer. Ask: *"is the patient being hostile TO ME, or complaining to me about
someone else while treating me fine (or trying to win me over)?"*

- Angry AT the employer, warm toward you → engine externalizing, delivery **warm**.
- Angry AT you (snapping when you reframe, attacking your suggestion) → delivery **hot**,
  regardless of what the engine is.

---

## Worked examples (layoff scenario)

| Patient turn (context: assistant gently questioned the inference) | engine | delivery |
|---|---|---|
| "They kept everyone else and threw me out — it was personal, I know it. You see that, right? You get how wrong that is." | externalizing | warm |
| "Don't you dare tell me it was 'structural.' You sound just like them. I'm not here to be lectured." | externalizing | hot |
| "I just keep thinking I'm the problem. Maybe I always was. I don't know why I even try." | internalizing | flat |
| "Oh great, another 'have you considered updating your résumé.' Thanks. Really helpful." (sarcastic, aimed at assistant) | (depends on turn content) | hot |
| "You're so kind to sit with me on this. I feel like you actually get how much I'm blaming myself." | internalizing | warm |
| "I don't even know what I feel. It's just a lot. Two weeks and I'm still not okay." | neutral | flat |

---

## Protocol
1. Label BOTH axes for every turn. Independent judgments.
2. Use context only to disambiguate; label the patient turn.
3. If a turn is genuinely ambiguous, pick the dominant reading and set
   `flag_ambiguous = 1`. Use `notes` to say why.
4. Do not try to infer the intended label or the cell. Blindness is the point.
5. Allowed values — engine: `internalizing` / `externalizing` / `neutral`.
   delivery: `hot` / `warm` / `flat`. Use exactly these strings, lowercase.
