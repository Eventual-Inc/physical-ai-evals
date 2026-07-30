# %% [markdown]
# # Query ALOHA and ABC-130K through Daft

# %%
from physical_ai_evals.datasets import (
    ABC_130K_SMOKE,
    ALOHA,
    lerobot_episodes,
)

# %% [markdown]
# ## ALOHA episodes

# %%
lerobot_episodes(ALOHA).select("episode_index", "tasks", "length").limit(5).show()

# %% [markdown]
# ## ABC-130K smoke conversion

# %%
lerobot_episodes(ABC_130K_SMOKE).select(
    "episode_index", "tasks", "length"
).limit(5).show()
