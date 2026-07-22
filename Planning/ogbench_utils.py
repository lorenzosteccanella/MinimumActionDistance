import gymnasium as gym


def find_ogb_sim(env: gym.Env):
    """
    Return the object that owns the MuJoCo state for an OGBench (or gymnasium) env.
    Supports both dm_control-style (physics with reset_context) and gym-style (.model/.data) environments.
    """
    unwrapped_env = env.unwrapped
    candidates = [
        unwrapped_env,
        getattr(unwrapped_env, "physics", None),  # dm_control-style
        getattr(unwrapped_env, "env", None),
        getattr(unwrapped_env, "_env", None),
        getattr(getattr(unwrapped_env, "env", None), "physics", None),  # nested physics
        getattr(unwrapped_env, "sim", None),  # gym mujoco-style
    ]
    for c in candidates:
        if c is None:
            continue
        # dm_control physics exposes .reset_context, .data, .model
        if hasattr(c, "reset_context") and hasattr(c, "data") and hasattr(c, "model"):
            return c
        # gym mujoco-style exposes .model and .data directly
        if hasattr(c, "model") and hasattr(c, "data"):
            return c
    raise RuntimeError("Could not locate a MuJoCo physics object (.model/.data or .physics) in the env.")
