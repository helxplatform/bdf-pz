if len(dataset_revisions) == 1:
    # This tool was used on the first revision of the dataset, i.e., either
    # immediately following set_input_dataset or backtracked to that point.
    raise ValueError(
        f"The dataset is already on the base revision: { ' '.join(dataset_revisions[0][1]) }. "
        "It cannot be backtracked any further."
    )

# Remove the current dataset from the revision list.
(current_dataset, removed_revision_action) = dataset_revisions.pop()
# Backtrack the revision
(dataset, backtracked_revision_action) = dataset_revisions[-1]
f"Removed operation: {' '.join(f'"{ i }"' for i in removed_revision_action)}\nCurrent operation: {' '.join(f'"{ i }"' for i in backtracked_revision_action)}"