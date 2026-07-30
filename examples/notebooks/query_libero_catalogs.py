# %% [markdown]
# # Query LIBERO-Para and LIBERO-Pro task plans

# %%
import daft

from physical_ai_evals import libero_para, libero_pro

# %% [markdown]
# ## LIBERO-Para variants

# %%
para = libero_para(episodes=1).specs
para.groupby("perturbation").agg(daft.col("bddl_path").count().alias("tasks")).sort(
    "perturbation"
).show()

# %%
para.where((daft.col("task_id") == 3) & (daft.col("perturbation") == "obj")).select(
    "task_key", "instruction", "bddl_path"
).limit(5).show()

# %% [markdown]
# ## LIBERO-Pro perturbations

# %%
pro = libero_pro("libero_spatial", episodes=1).specs
pro.groupby("perturbation").agg(daft.col("bddl_path").count().alias("tasks")).sort(
    "perturbation"
).show()

# %%
pro.where(daft.col("perturbation") == "lan").select("task_key", "bddl_path", "init_path").limit(
    5
).show()
