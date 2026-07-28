# %% [markdown]
# # Query ALOHA, EgoDex, and ABC-130K
#
# These queries read episode metadata from revision-checked LeRobot v3 datasets. They do
# not decode video frames.

# %%
from physical_ai_evals.datasets import abc, aloha, egodex

# %% [markdown]
# ## ALOHA episodes

# %%
aloha.episodes().select("episode_index", "tasks", "length").limit(5).show()

# %% [markdown]
# ## EgoDex activities and episodes

# %%
egodex.catalog(split="test").select("task_name", "dataset_uri").limit(10).show()

# %%
egodex.episodes("add_remove_lid", split="test").select(
    "episode_index", "tasks", "length"
).limit(5).show()

# %% [markdown]
# ## ABC-130K smoke conversion

# %%
abc.episodes(
    repo_id=abc.SMOKE_REPO_ID,
    revision=abc.SMOKE_REVISION,
).select("episode_index", "tasks", "length").limit(5).show()
