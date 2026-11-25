# ADR-0013: Learning order — roadmap and successors under policy learn-order-v1

- **Status:** Proposed
- **Date:** 2026-11-25
- **Deciders:** mentee team; mentors to accept
- **Mirrors:** none yet — mirror to `TA-memory/decisions/` on mentor acceptance

## Context

Two asks need an *order*, not only connections:

- A learner: "I am a data analyst and want to become a data scientist. What do I learn, and in
  what order?"
- A mentor: "Start from skill A. What comes after it, and after that?" That is a successor tree.

None of the five suites stores a complete "learn A before B" graph. A first reading of the link
names concluded that an order could not be had. A closer look at the data the links and nodes
carry finds seven official order signals already in our graph:

| Signal | Suite | Orders |
|---|---|---|
| Skill and knowledge **level** 0–7 per occupation, with importance | O*NET | how far each skill must go for a job |
| **Essential-for** links between skills (189) | ESCO | a stated prerequisite: skill Y needs X |
| **Knowledge vs skill/competence** | ESCO | the theory before its application |
| **Reuse level**: transversal, cross-sector, sector-specific, occupation-specific | ESCO | general before specific |
| **Levels 1–7** within a skill | SFIA | progression inside one skill |
| **Job Zones** 1–5 | O*NET | preparation a job needs |
| **Entry education, experience, training** | BLS | credentials before a job |

## Decision

Add a learning-order capability with **one named, versioned, deterministic policy:
`learn-order-v1`**. It follows the pattern of the Evaluator policies (ADR-0011). No model call.

**Roadmap(A → B)**, per suite where both A and B resolve with the usual clear-match rule:

- **Gap.**
  - ESCO: B's essential skills minus A's. JobTech: B's skills minus A's.
  - O*NET: B's skills and knowledge with importance ≥ 3.0 whose level for B exceeds A's by
    ≥ 0.5, keeping both levels.
  - BLS: B's entry education, experience and training, as a credentials block.
- **Order.**
  1. A stated ESCO essential-for link puts the needed skill first, using a deterministic
     topological order. A cycle is broken by node ID and reported.
  2. Then the step:
     - step 1 Foundations: knowledge;
     - step 2 Core: transversal and cross-sector skills, and O*NET skills;
     - step 3 Field: sector-specific;
     - step 4 Job-specific: occupation-specific.
  3. Within a step: the largest level jump first, then importance, then label, then node ID.
- **Each item** carries its step, suite, ID, citation, reason codes, and
  `stated_prerequisite: true|false`.

**Successors(X)**, a tree with depth, breadth and `max_nodes` caps:

- **Children of X:**
  - stated: skills that ESCO says need X; the next SFIA level of X;
  - policy: X's narrower ESCO skills; ESCO skills related to X with a later reuse level.
- **Order:** stated children first, then by reuse level, label and node ID.
- **Rules:** a node is never repeated, hubs are not expanded, and a suite with no order data
  reports `no_order_data` instead of guessing.

Every result prints its meaning: *"ordered by learn-order-v1 over official taxonomy data; only
items marked stated_prerequisite are prerequisites stated by a taxonomy"*.

## Consequences

- Learners and universities get an ordered, cited roadmap. Universities still decide their own
  prerequisites. The roadmap is evidence, not a syllabus.
- Stated prerequisites are sparse: 189 ESCO links, with only 26 two-step chains. Most of the
  order comes from the policy, and the output says so item by item.
- Changing the rules means a new policy name (`learn-order-v2`), never a silent change.
- If a suite later publishes real prerequisite edges, or a university's curriculum is loaded as
  its own graph, rule 1 takes them with no other change.
