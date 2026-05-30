"""Autonomy layer for Hermes.

A lightweight system that lets the agent act on its own initiative rather than
only in response to incoming messages or cron timers.

Pipeline (see the module docstrings for detail):

    collector.collect_state()   — zero-token state snapshot (no model calls)
        -> aux_model.decide_wake()   — one cheap model call: wake or not?
            -> session_spawn.spawn_autonomy_session()   — full agent session

The runner (``runner.maybe_run_autonomy_cycle``) ties these together and is
ticked by the gateway's cron ticker thread. response_log tracks what the agent
has already acted on so signals are not re-surfaced every cycle.
"""
