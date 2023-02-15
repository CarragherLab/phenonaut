# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

kinds = {
    "drugseq": {
        "transforms": [
            ("replace_str", ("Var2", "Pos_", "Pos-")),
            ("replace_str", ("Var2", "Neg_", "Neg-")),
            ("split_column", ("Var2", "_", ["WellName", "CpdID", "BARCODE"])),
            ("replace_str", "Var2", "Pos-", "Pos_"),
            ("replace_str", "Var2", "Neg-", "Neg_"),
            ("pivot", "Var1", "value"),
        ],
        "features_prefix": "ENSG",
        "index_col": 0,
    },
    "feat": {"features_prefix": "feat_"},
}
