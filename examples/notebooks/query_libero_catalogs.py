# %% [markdown]
# # Query LIBERO-Para and LIBERO-Pro task catalogs

# %%
import daft

from physical_ai_evals import libero_para_tasks, libero_pro_tasks

# %% [markdown]
# ## LIBERO-Para variants

# %%
para = libero_para_tasks()
para.groupby("perturbation").agg(
    daft.col("bddl_path").count().alias("tasks")
).sort("perturbation").show()

# %%
para.where(
    (daft.col("task_id") == 3) & (daft.col("perturbation") == "obj")
).select(
    "task_key", "paraphrase_key", "bddl_path"
).limit(5).show()

# %% [markdown]
# ## LIBERO-Pro perturbations

# %%
pro = libero_pro_tasks()
pro.groupby("suite", "perturbation").agg(
    daft.col("bddl_path").count().alias("tasks")
).sort(["suite", "perturbation"]).show()

# %%
pro.where(
    (daft.col("suite") == "libero_spatial")
    & (daft.col("perturbation") == "lan")
).select(
    "task_key", "bddl_path", "init_path"
).limit(5).show()
