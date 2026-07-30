# %% [markdown]
# # Query ALOHA, EgoDex, and ABC-130K through Daft

# %%
from physical_ai_evals.datasets import (
    ABC_130K_SMOKE,
    ALOHA,
    egodex,
    egodex_catalog,
    lerobot_episodes,
)

# %% [markdown]
# ## ALOHA episodes

# %%
lerobot_episodes(ALOHA).select("episode_index", "tasks", "length").limit(5).show()

# %% [markdown]
# ## EgoDex activities and episodes

# %%
egodex_catalog(split="test").select("task_name", "dataset_uri").limit(10).show()

# %%
lerobot_episodes(egodex("add_remove_lid", split="test")).select(
    "episode_index", "tasks", "length"
).limit(5).show()

# %% [markdown]
# ## ABC-130K smoke conversion

# %%
lerobot_episodes(ABC_130K_SMOKE).select(
    "episode_index", "tasks", "length"
).limit(5).show()
