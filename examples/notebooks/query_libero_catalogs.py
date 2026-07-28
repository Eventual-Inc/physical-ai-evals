# %% [markdown]
# # Query LIBERO task catalogs
#
# Catalog construction reads Hugging Face repository manifests. BDDL contents are read only
# for the filtered rows passed to `instructions()`.

# %%
import daft

from physical_ai_evals.datasets import libero_para, libero_pro

# %% [markdown]
# ## LIBERO-Para variants

# %%
para = libero_para.raw()
para.groupby("paraphrase_type").agg(
    daft.col("bddl_path").count().alias("tasks")
).sort("paraphrase_type").show()

# %%
para_sample = para.where(
    (daft.col("environment_task_id") == 3)
    & (daft.col("paraphrase_type") == "obj")
).limit(5)
libero_para.instructions(para_sample).select(
    "task_name", "paraphrase_key", "instruction"
).show()

# %% [markdown]
# ## LIBERO-PRO perturbations

# %%
pro = libero_pro.raw()
pro.groupby("suite", "perturbation").agg(
    daft.col("bddl_path").count().alias("tasks")
).sort(["suite", "perturbation"]).show()

# %%
pro_sample = pro.where(
    (daft.col("suite") == "libero_spatial")
    & (daft.col("perturbation") == "lan")
).limit(5)
libero_pro.instructions(pro_sample).select(
    "suite_variant", "task_name", "instruction"
).show()
