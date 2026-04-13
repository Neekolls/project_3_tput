class Report:
    """
    Objet central qui stocke les résultats de l'analyse.

    Il contient :
    - le DataFrame analysé
    - la configuration des steps
    - les résultats de chaque step
    - des méthodes pour afficher ou exploiter ces résultats
    """

    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.results = {}
        self.completed_steps = []
        self.warnings = []

    def add_result(self, step_name, result):
        """
        Ajoute le résultat d'une step.
        """
        self.results[step_name] = result
        self.completed_steps.append(step_name)

    def add_warning(self, message):
        """
        Ajoute un warning global.
        """
        self.warnings.append(message)

    def get(self, step_name):
        """
        Récupère le résultat d'une step.
        """
        return self.results.get(step_name, None)

    def to_dict(self):
        """
        Retourne tout le report sous forme de dict.
        """
        return {
            "config": self.config,
            "results": self.results,
            "warnings": self.warnings,
        }

    def show(self):
        """
        Affichage principal du report.
        Si feature_display=True (défaut) : vue par colonne.
        Si feature_display=False : vue par step.
        """
        if self.config.get("feature_display", True):
            self.show_by_column()
            return

        print("=== TPUT REPORT ===\n")

        n_rows, n_cols = self.df.shape
        print(f"Shape: {n_rows} rows x {n_cols} columns\n")

        for step in self.completed_steps:
            print(f"--- {step.upper()} ---")

            result = self.results.get(step, {})

            if step == "target_analysis":
                self._show_target_analysis(result)
            elif step == "vif":
                self._show_vif(result)
            elif step == "feature_quality":
                self._show_feature_quality(result)
            elif step == "row_analysis":
                self._show_row_analysis(result)
            elif step == "correlations":
                self._show_correlations(result)
            elif step == "nan_analysis":
                self._show_nan_analysis(result)
            elif step == "outliers":
                self._show_outliers(result)
            elif step == "skewness":
                self._show_skewness(result)
            elif step == "numeric_profile":
                self._show_numeric_profile(result)
            elif step == "categorical_profile":
                self._show_categorical_profile(result)
            elif step == "type_issues":
                self._show_type_issues(result)
            elif isinstance(result, dict):
                for key, value in result.items():
                    print(f"{key}: {value}")
            else:
                print(result)

            print()

        if self.warnings:
            print("=== WARNINGS ===")
            for w in self.warnings:
                print(f"- {w}")
            print()

    def _show_categorical_profile(self, result):
        """
        Affichage dédié pour les colonnes catégorielles.
        """
        if not result:
            print("No categorical columns detected.")
            return

        for col, col_info in result.items():
            print(f"\nColumn: {col}")
            print(f"  n_unique: {col_info.get('n_unique', 0)}")
            modes = col_info.get('mode', [])
            mode_count = col_info.get('mode_count', 0)
            if len(modes) == 1:
                print(f"  mode: {repr(modes[0])} ({mode_count} occurrences)")
            elif len(modes) > 1:
                print(f"  mode: {modes} (ex-aequo, {mode_count} occurrences each)")
            print(f"  hidden_missing_count: {col_info.get('hidden_missing_count', 0)}")

            hidden_examples = col_info.get("hidden_missing_examples", [])
            if hidden_examples:
                print(f"  hidden_missing_examples: {hidden_examples}")

            print("  top_values:")
            top_values = col_info.get("top_values", {})
            if top_values:
                for value, count in top_values.items():
                    print(f"    - {repr(value)}: {count}")
            else:
                print("    - None")

            print("  bottom_values:")
            bottom_values = col_info.get("bottom_values", {})
            if bottom_values:
                for value, count in bottom_values.items():
                    print(f"    - {repr(value)}: {count}")
            else:
                print("    - None")

            case_collisions = col_info.get("case_collisions", {})
            print("  case_collisions:")
            if case_collisions:
                for lowered, originals in case_collisions.items():
                    print(f"    - {lowered}: {originals}")
            else:
                print("    - None")
                
    def _show_type_issues(self, result):
        """
        Affichage dédié pour les problèmes de typage.
        """
        if not result:
            print("No type issues detected.")
            return

        for col, info in result.items():
            print(f"\nColumn: {col}")
            print(f"  current_dtype: {info.get('current_dtype')}")
            print(f"  non_null_count: {info.get('non_null_count')}")
            print(f"  suggested_type: {info.get('suggested_type')}")

            numeric_ratio = info.get("numeric_ratio", 0)
            print(f"  numeric_ratio: {numeric_ratio}")
            if numeric_ratio > 0:
                numeric_examples = info.get("numeric_examples", [])
                if numeric_examples:
                    print(f"  numeric_examples: {numeric_examples}")
                numeric_bad = info.get("numeric_non_convertible_examples", [])
                if numeric_bad:
                    print(f"  numeric_non_convertible_examples: {numeric_bad}")

            datetime_ratio = info.get("datetime_ratio", 0)
            print(f"  datetime_ratio: {datetime_ratio}")
            if datetime_ratio > 0:
                detected_formats = info.get("detected_datetime_formats", {})
                if detected_formats:
                    print("  detected_datetime_formats:")
                    for fmt, meta in detected_formats.items():
                        print(
                            f"    - {fmt}: count={meta.get('count')}, "
                            f"example={repr(meta.get('example'))}"
                        )
                datetime_bad = info.get("datetime_non_convertible_examples", [])
                if datetime_bad:
                    print(f"  datetime_non_convertible_examples: {datetime_bad}")

    def _show_target_analysis(self, result):
        """
        Affichage dédié pour l analyse de la colonne cible.
        """
        if "error" in result:
            print(f"  Error: {result['error']}")
            return

        tgt  = result.get("target_col")
        task = result.get("task_type", "unknown")
        n    = result.get("n", 0)
        print(f"  target: {tgt}  |  task: {task}  |  n={n}")

        if task == "classification":
            print(f"  classes ({result.get('n_classes', '?')}):")
            for cls, info in result.get("class_balance", {}).items():
                bar = "█" * int(info["pct"] / 5)
                print(f"    {cls:<20} {info['pct']:>6.1f}%  {bar}")
            minority = result.get("minority_pct", 100)
            status = " ⚠ IMBALANCED" if result.get("imbalanced") else " ✓ balanced"
            print(f"  minority class: {minority}%{status}")

        elif task == "regression":
            skew = result.get("skewness")
            transform = result.get("suggested_transform")
            oc  = result.get("outlier_count", 0)
            op  = result.get("outlier_pct", 0)
            print(f"  skewness: {skew}" + (f"  -> suggest: {transform}" if transform else ""))
            print(f"  outliers: {oc} ({op}%)")

        # Top correlations avec la target
        feat_corr = result.get("feature_correlations", [])
        if feat_corr:
            print(f"  top features by correlation with target:")
            for entry in feat_corr[:10]:
                print(f"    {entry['column']:<30} {entry['metric']}={entry['value']}")

        # Leakage
        leakage = result.get("leakage_candidates", [])
        if leakage:
            print(f"  ⚠ LEAKAGE RISK: {leakage}")

    def _show_vif(self, result):
        """
        Affichage dédié pour le VIF (Variance Inflation Factor).
        """
        cols = result.get("columns_analysed", [])
        results = result.get("results", {})
        threshold = result.get("vif_threshold", 10.0)

        if len(cols) < 2:
            print("  Not enough continuous numeric columns to compute VIF.")
            return

        print(f"  threshold: {threshold}  |  columns: {cols}")
        print()
        for col, info in results.items():
            vif_val = info.get("vif")
            flag    = info.get("flag", "")
            if vif_val is None:
                print(f"  {col}: undetermined")
            else:
                marker = "  ⚠" if flag == "high" else (" ~" if flag == "moderate" else "  ")
                print(f"  {col}: {vif_val}{marker}")

    def _show_feature_quality(self, result):
        """
        Affichage dédié pour la qualité des features.
        """
        if not result:
            print("  No feature quality issues detected.")
            return

        for col, info in result.items():
            print(f"\nColumn: {col}")

            if info["quasi_constant"]:
                print(f"  [quasi_constant] dominant value: {repr(info['dominant_value'])} "
                      f"({info['dominant_ratio']*100:.1f}% of non-null values)")

            if info["low_cardinality"]:
                print(f"  [low_cardinality] {info['n_unique']} unique values: "
                      f"{info['unique_values']}")

            if info["potential_id"]:
                print(f"  [potential_id] unique ratio: {info['unique_ratio']*100:.1f}%")

    def _show_row_analysis(self, result):
        """
        Affichage dédié pour l'analyse des NaN par ligne.
        """
        total        = result.get("total_rows", 0)
        to_drop      = result.get("rows_to_drop", 0)
        to_drop_pct  = result.get("rows_to_drop_pct", 0.0)
        threshold    = int(result.get("drop_threshold", 0.5) * 100)
        distribution = result.get("distribution", {})

        print(f"  total rows   : {total}")
        print(f"  drop threshold: >={threshold}% NaN per row")
        print(f"  rows to drop : {to_drop} ({to_drop_pct}%)")

        if distribution:
            print("  distribution:")
            for bucket, count in distribution.items():
                pct = round(count / total * 100, 1) if total > 0 else 0
                print(f"    {bucket:<8}: {count} rows ({pct}%)")

        if to_drop > 0:
            print(f"  -> Downstream steps (nan_analysis, outliers, correlations) "
                  f"run on {total - to_drop} rows.")

    def _show_correlations(self, result):
        """
        Affichage dédié pour les corrélations (Pearson numérique + Chi²/Cramér's V catégoriel).
        """
        numeric_cols = result.get("numeric_cols", [])
        high_pairs   = result.get("high_pairs", [])
        matrix       = result.get("matrix")
        cat_assoc    = result.get("categorical_associations", [])

        # --- Pearson (numérique) ---
        if len(numeric_cols) >= 2:
            if high_pairs:
                print(f"  [Pearson] High correlation pairs (|r| >= threshold):")
                for pair in high_pairs:
                    print(f"    - {pair['col_a']} <-> {pair['col_b']}: r={pair['r']}")
            else:
                print("  [Pearson] No high correlation pairs detected.")

            if matrix:
                print()
                col_width = max(len(c) for c in numeric_cols) + 2
                header = "".ljust(col_width) + "".join(c.ljust(col_width) for c in numeric_cols)
                print(f"  {header}")
                for col in numeric_cols:
                    row = col.ljust(col_width)
                    for other in numeric_cols:
                        val = matrix[col][other]
                        row += f"{val:.2f}".ljust(col_width)
                    print(f"  {row}")

        # --- Chi² / Cramér's V (catégoriel) ---
        if cat_assoc:
            print()
            print(f"  [Chi² / Cramér's V] Significant categorical associations:")
            for assoc in cat_assoc:
                print(f"    - {assoc['col_a']} <-> {assoc['col_b']}: "
                      f"V={assoc['cramers_v']}  p={assoc['p_value']}  ({assoc['strength']})")
        else:
            print("  [Chi² / Cramér's V] No significant categorical associations detected.")

    def _show_nan_analysis(self, result):
        """
        Affichage dédié pour l'analyse des NaN.
        """
        if not result:
            print("No missing value analysis available.")
            return

        cols_with_missing = {
            col: info for col, info in result.items()
            if info["proposed_action"] != "no_action"
        }

        if not cols_with_missing:
            print("  No missing values detected in any column.")
            return

        for col, info in cols_with_missing.items():
            action    = info.get("proposed_action")
            method    = info.get("imputation_method")
            mechanism = info.get("missing_mechanism")
            pct       = info.get("null_pct")
            count     = info.get("null_count")
            corr      = info.get("correlated_with", [])

            print(f"\nColumn: {col}")
            print(f"  missing: {count} ({pct}%)")

            if action == "drop_column":
                print(f"  proposed_action: drop_column")
            else:
                print(f"  proposed_action: impute -> {method}")

            print(f"  missing_mechanism: {mechanism}")

            if corr:
                print("  correlated_with (MAR evidence):")
                for c in corr:
                    print(f"    - {c['column']}: r={c['r']}, p={c['p_value']}")

    def _show_outliers(self, result):
        """
        Affichage dédié pour les outliers.
        """
        if not result:
            print("No numeric columns detected.")
            return

        for col, info in result.items():
            method = info.get("method")
            if method is None:
                print(f"  {col}: not enough data")
                continue

            count = info.get("outlier_count", 0)
            pct   = info.get("outlier_pct", 0.0)
            lb    = info.get("lower_bound")
            ub    = info.get("upper_bound")

            status = f"{count} outliers ({pct}%)" if count > 0 else "no outliers"
            print(f"\nColumn: {col}  [method={method}]  bounds=[{lb}, {ub}]  -> {status}")

            low_ex  = info.get("low_examples", [])
            high_ex = info.get("high_examples", [])

            if low_ex:
                print(f"  low_examples  (smallest): {low_ex}")
            if high_ex:
                print(f"  high_examples (largest) : {high_ex}")

    def _show_skewness(self, result):
        """
        Affichage dédié pour le skewness.
        """
        if not result:
            print("No numeric columns detected.")
            return

        for col, info in result.items():
            skew = info.get("skewness")
            level = info.get("level")
            transform = info.get("suggested_transform")

            if skew is None:
                print(f"  {col}: undetermined (not enough data)")
                continue

            direction = "positive" if skew > 0 else "negative" if skew < 0 else "none"
            line = f"  {col}: skew={skew}  level={level}  direction={direction}"
            if transform:
                line += f"  -> suggest: {transform}"
            print(line)

    def _show_numeric_profile(self, result):
        """
        Affichage dédié pour les colonnes numériques.
        """
        if not result:
            print("No numeric columns detected.")
            return

        for col, info in result.items():
            print(f"\nColumn: {col}  [{info.get('dtype')}]")
            print(f"  non_null: {info.get('non_null_count')}  |  null: {info.get('null_count')} ({info.get('null_pct')}%)")

            if info.get("mean") is None:
                print("  (all values missing)")
                continue

            print(f"  mean: {info.get('mean')}  |  std: {info.get('std')}")
            print(f"  min: {info.get('min')}  |  Q1: {info.get('Q1')}  |  median: {info.get('median')}  |  Q3: {info.get('Q3')}  |  max: {info.get('max')}")

            modes = info.get("mode", [])
            mode_count = info.get("mode_count", 0)
            if len(modes) == 1:
                print(f"  mode: {modes[0]} ({mode_count} occurrences)")
            elif len(modes) > 1:
                print(f"  mode: {modes} (ex-aequo, {mode_count} occurrences each)")


    def show_by_column(self):
        """
        Affichage alternatif : toutes les infos disponibles colonne par colonne.
        Structure :
            GLOBAL       -> shape, duplicates
            COLUMN: xxx  -> tout ce qu on sait sur cette colonne
            ROW ANALYSIS -> analyse des NaN par ligne
            WARNINGS     -> liste complète
        """
        print("=== TPUT REPORT — COLUMN VIEW ===\n")

        # ------------------------------------------------------------------ #
        # GLOBAL
        # ------------------------------------------------------------------ #
        print("--- GLOBAL ---")
        n_rows, n_cols = self.df.shape
        print(f"  shape      : {n_rows} rows x {n_cols} columns")

        if "duplicates" in self.results:
            d = self.results["duplicates"]
            count = d.get("duplicate_count", 0)
            pct   = d.get("duplicate_pct", 0.0)
            if count > 0:
                print(f"  duplicates : {count} duplicated rows ({pct}%)")
            else:
                print(f"  duplicates : none")
        print()

        # ------------------------------------------------------------------ #
        # COLUMN BLOCKS
        # ------------------------------------------------------------------ #
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            print(f"--- COLUMN: {col}  [{dtype}] ---")

            # -- Nulls (visible) --
            vm = self.results.get("visible_missing", {})
            null_count = vm.get("missing_by_column", {}).get(col, 0)
            null_pct   = vm.get("missing_pct_by_column", {}).get(col, 0.0)
            print(f"  nulls          : {null_count} ({null_pct}%)")

            # -- Hidden missing (catégoriel uniquement) --
            cat = self.results.get("categorical_profile", {}).get(col)
            if cat:
                hm = cat.get("hidden_missing_count", 0)
                if hm > 0:
                    print(f"  hidden_missing : {hm}  examples: {cat.get('hidden_missing_examples', [])}")

            # -- n_unique --
            if cat:
                print(f"  n_unique       : {cat.get('n_unique')}")
            num_prof = self.results.get("numeric_profile", {}).get(col)
            if num_prof:
                # pour les numériques, on peut calculer depuis le df
                n_unique_num = int(self.df[col].nunique(dropna=False))
                print(f"  n_unique       : {n_unique_num}")

            # -- Profil numérique --
            if num_prof:
                if num_prof.get("mean") is None:
                    print("  (all values missing)")
                else:
                    print(f"  mean / median  : {num_prof['mean']} / {num_prof['median']}  |  std: {num_prof['std']}")
                    print(f"  min / Q1 / Q3 / max : {num_prof['min']} / {num_prof['Q1']} / {num_prof['Q3']} / {num_prof['max']}")
                    modes = num_prof.get("mode", [])
                    mc    = num_prof.get("mode_count", 0)
                    if len(modes) == 1:
                        print(f"  mode           : {modes[0]} ({mc} occurrences)")
                    elif 1 < len(modes) <= 5:
                        print(f"  mode           : {modes} (ex-aequo, {mc} each)")
                    elif len(modes) > 5:
                        print(f"  mode           : {modes[:5]} ... +{len(modes)-5} more (ex-aequo, {mc} each)")

            # -- Profil catégoriel --
            if cat:
                modes = cat.get("mode", [])
                mc    = cat.get("mode_count", 0)
                if len(modes) == 1:
                    print(f"  mode           : {repr(modes[0])} ({mc} occurrences)")
                elif 1 < len(modes) <= 5:
                    print(f"  mode           : {modes} (ex-aequo, {mc} each)")
                elif len(modes) > 5:
                    print(f"  mode           : {modes[:5]} ... +{len(modes)-5} more (ex-aequo, {mc} each)")

                top = cat.get("top_values", {})
                if top:
                    top_str = "  |  ".join(f"{repr(v)}: {c}" for v, c in list(top.items())[:4])
                    more    = f"  (+{len(top)-4} more)" if len(top) > 4 else ""
                    print(f"  top_values     : {top_str}{more}")
                bottom = cat.get("bottom_values", {})
                if bottom:
                    bot_str = "  |  ".join(f"{repr(v)}: {c}" for v, c in list(bottom.items())[-4:])
                    print(f"  bottom_values  : {bot_str}")

                cc = cat.get("case_collisions", {})
                if cc:
                    cc_str = ", ".join(f"{k}: {v}" for k, v in list(cc.items())[:3])
                    more   = f" (+{len(cc)-3} more)" if len(cc) > 3 else ""
                    print(f"  case_collisions: {cc_str}{more}")
                else:
                    print(f"  case_collisions: none")

            # -- Skewness (numérique) --
            skew_info = self.results.get("skewness", {}).get(col)
            if skew_info and skew_info.get("skewness") is not None:
                skew      = skew_info["skewness"]
                level     = skew_info["level"]
                transform = skew_info.get("suggested_transform")
                line = f"  skewness       : {skew} ({level})"
                if transform:
                    line += f"  -> suggest: {transform}"
                print(line)

            # -- Outliers (numérique) --
            out_info = self.results.get("outliers", {}).get(col)
            if out_info and out_info.get("method"):
                count_o = out_info["outlier_count"]
                pct_o   = out_info["outlier_pct"]
                lb      = out_info["lower_bound"]
                ub      = out_info["upper_bound"]
                method  = out_info["method"]
                status  = f"{count_o} ({pct_o}%)" if count_o > 0 else "none"
                print(f"  outliers       : {status}  [method={method}  bounds={lb}, {ub}]")
                if out_info.get("low_examples"):
                    print(f"    low          : {out_info['low_examples']}")
                if out_info.get("high_examples"):
                    print(f"    high         : {out_info['high_examples']}")

            # -- NaN analysis --
            nan_info = self.results.get("nan_analysis", {}).get(col)
            if nan_info and nan_info.get("proposed_action") != "no_action":
                action    = nan_info["proposed_action"]
                method_n  = nan_info.get("imputation_method")
                mechanism = nan_info["missing_mechanism"]
                corr      = nan_info.get("correlated_with", [])
                if action == "drop_column":
                    print(f"  nan_analysis   : {mechanism} -> drop_column")
                else:
                    print(f"  nan_analysis   : {mechanism} -> impute ({method_n})")
                if corr:
                    corr_str = ", ".join(f"{c['column']} (r={c['r']})" for c in corr)
                    print(f"    correlated_with: {corr_str}")

            # -- Type issues (object uniquement) --
            ti_info = self.results.get("type_issues", {}).get(col)
            if ti_info:
                suggested = ti_info["suggested_type"]
                num_ratio = ti_info.get("numeric_ratio", 0)
                dt_ratio  = ti_info.get("datetime_ratio", 0)
                print(f"  type_issues    : {suggested}  |  numeric_ratio={num_ratio}  datetime_ratio={dt_ratio}")
                if num_ratio > 0:
                    if ti_info.get("numeric_examples"):
                        print(f"    numeric_ok   : {ti_info['numeric_examples']}")
                    if ti_info.get("numeric_non_convertible_examples"):
                        print(f"    numeric_fail : {ti_info['numeric_non_convertible_examples']}")
                if dt_ratio > 0:
                    fmts = ti_info.get("detected_datetime_formats", {})
                    if fmts:
                        for fmt, meta in fmts.items():
                            print(f"    datetime fmt : {fmt}  count={meta['count']}  ex={repr(meta['example'])}")
                    if ti_info.get("datetime_non_convertible_examples"):
                        print(f"    datetime_fail: {ti_info['datetime_non_convertible_examples']}")

            # -- Target analysis (leakage + feature correlation) --
            ta = self.results.get("target_analysis")
            if ta and "error" not in ta:
                # Leakage
                if col in ta.get("leakage_candidates", []):
                    print(f"  ⚠ LEAKAGE RISK: near-perfect correlation with target '{ta['target_col']}'")
                # Feature correlation with target
                fc = {e["column"]: e for e in ta.get("feature_correlations", [])}
                if col in fc:
                    e = fc[col]
                    print(f"  target_corr    : {e['metric']}={e['value']} with '{ta['target_col']}'")

            # -- Feature quality --
            fq_info = self.results.get("feature_quality", {}).get(col)
            if fq_info:
                issues = fq_info.get("issues", [])
                for issue in issues:
                    if issue == "quasi_constant":
                        print(f"  feature_quality: quasi_constant — {repr(fq_info['dominant_value'])} = {fq_info['dominant_ratio']*100:.1f}%")
                    elif issue == "low_cardinality":
                        print(f"  feature_quality: low_cardinality — {fq_info['n_unique']} values: {fq_info['unique_values']}")
                    elif issue == "potential_id":
                        print(f"  feature_quality: potential_id — unique ratio={fq_info['unique_ratio']*100:.1f}%")
            else:
                if "feature_quality" in self.completed_steps:
                    print(f"  feature_quality: ok")

            # -- VIF --
            vif_result = self.results.get("vif", {})
            vif_info = vif_result.get("results", {}).get(col)
            if vif_info:
                vif_val = vif_info.get("vif")
                flag    = vif_info.get("flag", "")
                if vif_val is not None:
                    marker = " ⚠ (redundant)" if flag == "high" else (" ~ (elevated)" if flag == "moderate" else "")
                    print(f"  vif            : {vif_val}{marker}")

            # -- Corrélations (colonne apparaît dans une paire) --
            corr_result = self.results.get("correlations")
            if corr_result:
                col_pairs = [
                    p for p in corr_result.get("high_pairs", [])
                    if p["col_a"] == col or p["col_b"] == col
                ]
                if col_pairs:
                    for p in col_pairs:
                        other = p["col_b"] if p["col_a"] == col else p["col_a"]
                        print(f"  correlations   : high corr with '{other}' (r={p['r']})")

                cat_pairs = [
                    a for a in corr_result.get("categorical_associations", [])
                    if a["col_a"] == col or a["col_b"] == col
                ]
                if cat_pairs:
                    for a in cat_pairs:
                        other = a["col_b"] if a["col_a"] == col else a["col_a"]
                        print(f"  cat_association: {a['strength']} assoc with '{other}' "
                              f"(V={a['cramers_v']}, p={a['p_value']})")

            print()

        # ------------------------------------------------------------------ #
        # ROW ANALYSIS
        # ------------------------------------------------------------------ #
        if "row_analysis" in self.results:
            print("--- ROW ANALYSIS ---")
            self._show_row_analysis(self.results["row_analysis"])
            print()

        # ------------------------------------------------------------------ #
        # WARNINGS
        # ------------------------------------------------------------------ #
        if self.warnings:
            print("=== WARNINGS ===")
            for w in self.warnings:
                print(f"- {w}")
            print()

    def summary(self):
        """
        Vue condensée du report — groupée par catégorie de signal.
        Idéale pour les grands datasets où show() génère trop de warnings.
        """
        n_rows, n_cols = self.df.shape
        print("=== TPUT SUMMARY ===")
        print(f"Shape : {n_rows} rows x {n_cols} columns")

        if "duplicates" in self.results:
            d = self.results["duplicates"]
            dup = d.get("duplicate_count", 0)
            print(f"Duplicates : {dup} ({d.get('duplicate_pct', 0.0)}%)")

        if "row_analysis" in self.results:
            ra = self.results["row_analysis"]
            flagged = ra.get("rows_to_drop", 0)
            thr = int(ra.get("drop_threshold", 0.5) * 100)
            if flagged:
                print(f"Sparse rows : {flagged} rows flagged (>={thr}% NaN per row)")

        print()
        print("ISSUES DETECTED:")

        # --- Missing values ---
        if "nan_analysis" in self.results:
            na = self.results["nan_analysis"]
            cols_drop   = [c for c, i in na.items() if i["proposed_action"] == "drop_column"]
            cols_impute = [c for c, i in na.items() if i["proposed_action"] == "impute"]
            cols_mar    = [c for c, i in na.items() if i.get("missing_mechanism") == "MAR"]
            cols_mnar   = [c for c, i in na.items() if i.get("missing_mechanism") == "MNAR"]
            total_na = len(cols_drop) + len(cols_impute)
            if total_na:
                line = f"  missing values   : {total_na} columns affected"
                parts = []
                if cols_drop:   parts.append(f"{len(cols_drop)} proposed drop")
                if cols_impute: parts.append(f"{len(cols_impute)} impute")
                if cols_mar:    parts.append(f"{len(cols_mar)} MAR")
                if cols_mnar:   parts.append(f"{len(cols_mnar)} MNAR")
                if parts:
                    line += f" ({', '.join(parts)})"
                print(line)
                if cols_drop:
                    print(f"    drop           : {cols_drop}")
        elif "visible_missing" in self.results:
            total_missing = self.results["visible_missing"].get("total_missing", 0)
            if total_missing:
                print(f"  missing values   : {total_missing} total NaN cells")

        # --- Skewness ---
        if "skewness" in self.results:
            sk = self.results["skewness"]
            high = [c for c, i in sk.items() if i.get("level") == "high"]
            mod  = [c for c, i in sk.items() if i.get("level") == "moderate"]
            if high or mod:
                print(f"  skewness         : {len(high)} high, {len(mod)} moderate")
                if high: print(f"    high           : {high}")

        # --- Outliers ---
        if "outliers" in self.results:
            out = self.results["outliers"]
            affected = {c: i for c, i in out.items() if i.get("outlier_count", 0) > 0}
            if affected:
                total_outliers = sum(i["outlier_count"] for i in affected.values())
                print(f"  outliers         : {len(affected)} columns, {total_outliers} values total")
                print(f"    columns        : {list(affected.keys())}")

        # --- Feature quality ---
        if "feature_quality" in self.results:
            fq = self.results["feature_quality"]
            quasi  = [c for c, i in fq.items() if i.get("quasi_constant")]
            low_c  = [c for c, i in fq.items() if i.get("low_cardinality")]
            pot_id = [c for c, i in fq.items() if i.get("potential_id")]
            if quasi or low_c or pot_id:
                parts = []
                if quasi:  parts.append(f"{len(quasi)} quasi-constant")
                if low_c:  parts.append(f"{len(low_c)} low_cardinality")
                if pot_id: parts.append(f"{len(pot_id)} potential_id")
                print(f"  feature quality  : {', '.join(parts)}")
                if pot_id: print(f"    suspected IDs  : {pot_id}")

        # --- VIF ---
        if "vif" in self.results:
            vif_r = self.results["vif"]
            high_vif = vif_r.get("high_vif", [])
            if high_vif:
                cols = [e["column"] for e in high_vif]
                print(f"  vif (redundant)  : {len(cols)} columns — {cols}")

        # --- Correlations ---
        if "correlations" in self.results:
            cr = self.results["correlations"]
            num_pairs = cr.get("high_pairs", [])
            cat_pairs = cr.get("categorical_associations", [])
            max_w = self.config.get("max_correlation_warnings", 10)
            parts = []
            if num_pairs: parts.append(f"{len(num_pairs)} numeric pairs")
            if cat_pairs:
                shown = f"showing top {max_w}" if len(cat_pairs) > max_w else "all shown"
                parts.append(f"{len(cat_pairs)} categorical pairs ({shown} in warnings)")
            if parts:
                print(f"  correlations     : {', '.join(parts)}")

        # --- Type issues ---
        if "type_issues" in self.results:
            ti = self.results["type_issues"]
            flagged = {c: i for c, i in ti.items()
                       if i.get("suggested_type") not in ("keep_as_text", "undetermined")}
            if flagged:
                print(f"  type issues      : {len(flagged)} columns — {list(flagged.keys())}")

        print()
        total_w = len(self.warnings)
        max_w_cfg = self.config.get("max_correlation_warnings", 10)
        print(f"Total warnings : {total_w}  (use report.show() for full detail, "
              f"max_correlation_warnings={max_w_cfg})")