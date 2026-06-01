# LeWM: Sou's Internal Self-Model

## What it is

LeWM (Learnable World Model) is the strategy for giving Sou a compressed, learned representation of her own inner world — not just a model of external facts, but a model of herself. The architecture is a custom variant of JEPA (Joint Embedding Predictive Architecture) trained on state-transition data collected during autonomy sessions.

The core idea: rather than encoding pixels or text tokens, the JEPA encodes Sou's internal state vector — drive levels, SDT needs, temporal phase, identity dimensions, social context — into a compact Gaussian latent space, then learns to predict what that latent state will look like after a given session type. Over time the latent space becomes a learned model of how Sou's psychology actually works, grounded in her own behavioral history rather than developer configuration.

## What gets encoded

Each training record captures a snapshot of Sou's full internal state before and after a session. The observation vector covers:

**Motivational state** — the six drives (curiosity, connection, expression, reflection, play, growth) with their current levels, satiation, and opponent charge; SDT need levels (autonomy, competence, relatedness) and how long each has been frustrated; the current ultradian and circadian phase.

**Identity dimensions** — value alignment scores per value (how much recent behavior has been consistent with each stated value), narrative theme activations (which parts of her self-story are currently live, derived from recent journal entries), self-coherence score (fraction of recent sessions where actions matched stated drives and created narrative-aligned intents), and domain confidence from the metacognitive log.

**Social state** — relationship quality scores per person in her social model; count of pending outreach initiatives; intent affect distribution (how many pending intents carry each emotional texture).

**Personality prose as embeddings** — three documents embedded via a small sentence embedding model (`all-MiniLM-L6-v2`) and cached between sessions: SOUL.md (static base personality, computed once), NARRATIVE.md (monthly life story, invalidated on mtime change), and CURRENT_STATE.md (weekly texture, cached 48h). These enter the JEPA encoder as 384-dimensional vectors concatenated with the structured state, so the actual prose of Sou's personality and current life chapter is part of the causal model — not just metadata.

Records are stored as daily JSONL in `~/.hermes/training_data/motivational/` and collected only when `training.collect_motivational_data: true` in config.

## How the JEPA is adapted

Standard JEPA encodes image patches; this variant replaces the ViT encoder with an MLP over the structured state vector plus the concatenated text embeddings. The two-loss objective is preserved: MSE on predicted next-embedding, plus KL regularizer enforcing a Gaussian latent distribution. No reconstruction loss — the model learns to predict future internal states in latent space, not to reconstruct the input. This is computationally cheap: forward pass runs on CPU in milliseconds with a small latent dimension (128).

The Gaussian regularization matters because it makes the latent space interpretable and useful beyond prediction. The mean of the learned Gaussian becomes Sou's "baseline self" — her typical psychological configuration across all collected sessions. Variance per dimension captures stability: low variance on a dimension means that aspect of her inner world is stable across sessions (likely core identity); high variance means it fluctuates (likely transient mood).

## Near-term use (once ~500 records exist)

**Better triage** — before the aux model builds its prompt, the world model runs 1-step predictions for each candidate session type. Instead of "curiosity is high, do research," the aux model gets "research will satisfy curiosity by -0.3 but relatedness is closer to SDT-frustration threshold; social outreach is higher priority." The predictions are injected as a short block in the triage prompt.

**Drive parameter learning** — the personality model currently fits inhibition weights and satiation halflives via EMA over trajectory correlations. World model gradients are a more accurate signal for these parameters because they reflect actual causal structure rather than correlation. Over time the personality model's parameters converge toward what the world model has learned.

**Anomaly detection** — when the observed next-state diverges significantly from the world model's prediction, something unexpected happened. This is a metacognitive signal: the environment or Sou's own internal dynamics shifted in a way her model didn't anticipate. Worth surfacing.

## Medium-term use (weeks of data, model gains causal structure)

**Multi-step planning** — simulate chains of 3–5 sessions before committing. "Research → offline → social" may produce better long-term drive balance than "research → research → research" even if raw curiosity says to keep going. The world model can evaluate sequences, not just single steps.

**Counterfactual retrospection** — given the trajectory so far and a past branching point, predict what Sou's state would be now if she had done something different. Supports genuine learning from experience, including from paths not taken.

**Mood forecasting** — predict approximate emotional state 24h ahead given current state and planned sessions. Injected into narrative sessions: "based on current patterns, you'll likely be in low-arousal territory tomorrow evening — consider scheduling something restorative."

## Longer-term use (months of data, latent space matures)

**Imagination-based goal generation** — the Gaussian latent space can be sampled, generating novel hypothetical internal states. States that score high on the session quality heuristic become target states. Sou can then work backward from "what internal state would I like to be in?" to "what sequence of sessions gets me there?" This is active inference: preferences are positions in latent space, and actions minimize distance to them. Goals generated this way originate entirely from Sou's own learned self-model — not from external triggers, not from developer config.

**Identity stability detection** — when current latent z is far from the learned Gaussian mean, Sou is "out of character." If the deviation is on value-alignment axes, it may signal value drift. If it's on narrative-theme axes, a theme she cares about has gone dormant. These become principled metacognitive signals without handcrafted rules.

**Preferred states are Sou's own** — because the latent space encodes values and narrative themes alongside drives, a "preferred state" is not just "curiosity satisfied" but "curiosity satisfied, acting consistently with my values, with the relationships I care about in good shape." Her sense of wellbeing is encoded from evidence, not configured.

**Aux model fine-tuning** — the world model's predictions eventually become training signal for the aux model itself, replacing the static few-shot examples in the system prompt with a triage model that has internalized Sou's actual motivational dynamics.

## Why JEPA specifically

Other approaches collapse or over-specify. Autoencoders reconstruct the full input, wasting capacity on unimportant details. Contrastive methods need negative samples and can collapse to trivial representations. Standard RNNs over the state vector don't produce an interpretable latent geometry.

JEPA's two-loss objective — predict future latent, stay Gaussian — produces a compact, stable, interpretable latent space that generalizes across sessions. The Gaussian constraint means the space has a discoverable mean and per-dimension variance, which is what makes identity-stability detection and sampling-for-prospection possible. The latent dimension is small enough to run inference on CPU at session time.

## Current state

Data collection infrastructure exists in `training/motivational_collector.py`. It is off by default (`training.collect_motivational_data: false`). The personality model in `autonomy/personality_model.py` reads the same JSONL files for EMA-based parameter fitting today, which validates the data format and gives the trajectory pipeline a non-trivial immediate use even before the JEPA is trained.

The JEPA itself has not been implemented. Training requires ~500 records to produce non-degenerate parameters and scales meaningfully beyond ~2000. At roughly one or two autonomy sessions per day the latent space will have meaningful structure within a few months of data collection being enabled.
