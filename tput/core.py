import warnings
import pandas as pd
from datetime import datetime
from .report import Report

COMMON_DATETIME_FORMATS = [
    "%Y-%m-%d",      # 2024-05-07
    "%d-%m-%Y",      # 07-05-2024
    "%m-%d-%Y",      # 05-07-2024
    "%d.%m.%Y",      # 07.05.2024
    "%m.%d.%Y",      # 05.07.2024
    "%Y.%m.%d",      # 2024.05.07
    "%d/%m/%Y",      # 07/05/2024
    "%m/%d/%Y",      # 05/07/2024
    "%Y/%m/%d",      # 2024/05/07
    "%d %b %Y",      # 7 May 2024 si %b=May abrégé ? non, May marche souvent pas; garder aussi %B
    "%d %B %Y",      # 7 May 2024
    "%b %d %Y",      # May 7 2024
    "%B %d %Y",      # May 7 2024
    "%d %b, %Y",     # 7 May, 2024
    "%d %B, %Y",     # 7 May, 2024
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]

def _clean_string_value(x):
    """
    Nettoyage minimal pour les colonnes texte.
    """
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return pd.NA
    return x

def _detect_datetime_formats(series, formats=None):
    """
    Détecte les formats datetime connus présents dans une série texte.

    Parameters
    ----------
    series : pandas.Series
        Série déjà nettoyée, sans NaN conseillés
    formats : list[str] | None
        Liste de formats datetime à tester

    Returns
    -------
    dict
        {
            "datetime_convertible_count": int,
            "datetime_ratio": float,
            "detected_formats": {
                "%d.%m.%Y": {
                    "count": int,
                    "example": str
                },
                ...
            },
            "non_convertible_examples": list[str]
        }
    """
    if formats is None:
        formats = COMMON_DATETIME_FORMATS

    non_null = series.dropna()
    total = len(non_null)

    if total == 0:
        return {
            "datetime_convertible_count": 0,
            "datetime_ratio": 0.0,
            "detected_formats": {},
            "non_convertible_examples": [],
        }

    detected_formats = {}
    convertible_indices = set()
    non_convertible_examples = []

    for idx, value in non_null.items():
        if not isinstance(value, str):
            value = str(value)

        matched = False

        for fmt in formats:
            try:
                datetime.strptime(value, fmt)
                matched = True
                convertible_indices.add(idx)

                if fmt not in detected_formats:
                    detected_formats[fmt] = {
                        "count": 0,
                        "example": value,
                    }

                detected_formats[fmt]["count"] += 1
                break

            except ValueError:
                continue

        if not matched and len(non_convertible_examples) < 5:
            non_convertible_examples.append(value)

    datetime_convertible_count = len(convertible_indices)
    datetime_ratio = round(datetime_convertible_count / total, 2)

    return {
        "datetime_convertible_count": datetime_convertible_count,
        "datetime_ratio": datetime_ratio,
        "detected_formats": detected_formats,
        "non_convertible_examples": non_convertible_examples,
    }

def compute_overview(df):
    """
    Donne une vue d'ensemble du DataFrame.
    """
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def compute_visible_missing(df):
    """
    Détecte les NaN 'réels' dans le DataFrame.
    """
    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())
    missing_by_column = missing_counts.to_dict()

    n_rows = len(df)
    missing_pct_by_column = {
        col: round((count / n_rows) * 100, 2) if n_rows > 0 else 0
        for col, count in missing_by_column.items()
    }

    return {
        "total_missing": total_missing,
        "missing_by_column": missing_by_column,
        "missing_pct_by_column": missing_pct_by_column,
    }



def compute_categorical_profile(df, top_n=20):
    """
    Analyse les colonnes catégorielles / texte.

    Pour chaque colonne texte :
    - nombre de valeurs uniques
    - top N des valeurs les plus fréquentes
    - hidden missing ("", "   ", etc.)
    - collisions de casse ("Paris" vs "paris")
    """
    result = {}

    for col in df.columns:
        series = df[col]

        if not (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
            continue

        n_unique = int(series.nunique(dropna=False))
        vc = series.value_counts(dropna=False)
        top_values    = vc.head(top_n).to_dict()
        bottom_values = vc.tail(top_n).to_dict()

        hidden_mask = series.apply(
            lambda x: isinstance(x, str) and x.strip() == ""
        )
        hidden_missing_count = int(hidden_mask.sum())
        hidden_missing_examples = pd.Series(series[hidden_mask].unique()).tolist()

        normalized_map = {}
        for value in series.dropna().unique():
            if isinstance(value, str):
                lowered = value.strip().lower()
                normalized_map.setdefault(lowered, set()).add(value)

        case_collisions = {
            lowered: sorted(list(originals))
            for lowered, originals in normalized_map.items()
            if len(originals) > 1
        }

        value_counts = series.value_counts(dropna=True)
        if not value_counts.empty:
            max_count = value_counts.iloc[0]
            modes = value_counts[value_counts == max_count].index.tolist()
            mode_count = int(max_count)
        else:
            modes = []
            mode_count = 0

        result[col] = {
            "n_unique": n_unique,
            "top_values": top_values,
            "bottom_values": bottom_values,
            "mode": modes,
            "mode_count": mode_count,
            "hidden_missing_count": hidden_missing_count,
            "hidden_missing_examples": hidden_missing_examples,
            "case_collisions": case_collisions,
        }

    return result

def compute_duplicates(df, sample_size=5):
    """
    Détecte les lignes dupliquées dans le DataFrame.
    """

    duplicated_mask = df.duplicated()
    duplicate_count = int(duplicated_mask.sum())

    n_rows = len(df)
    duplicate_pct = round((duplicate_count / n_rows) * 100, 2) if n_rows > 0 else 0

    # On prend quelques exemples de doublons
    duplicate_examples = df[duplicated_mask].head(sample_size).to_dict(orient="records")

    return {
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "duplicate_examples": duplicate_examples,
    }

def compute_type_issues(df, threshold=0.8, possible_threshold=0.5):
    """
    Détecte les colonnes texte qui pourraient être mal typées.

    suggested_type possibles :
    - "numeric"                       : ratio >= threshold, clairement numérique
    - "possible_numeric"              : ratio entre possible_threshold et threshold
    - "datetime"                      : ratio >= threshold, un seul format détecté
    - "datetime_mixed_formats"        : ratio >= threshold, plusieurs formats détectés
    - "possible_datetime"             : ratio entre possible_threshold et threshold, un format
    - "possible_datetime_mixed_formats" : ratio entre possible_threshold et threshold, plusieurs formats
    - "ambiguous"                     : à la fois numeric et datetime >= threshold
    - "keep_as_text"                  : rien de significatif détecté
    - "undetermined"                  : colonne entièrement vide
    """
    result = {}

    for col in df.columns:
        series = df[col]

        if not (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
            continue

        cleaned = series.apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        cleaned = cleaned.replace("", pd.NA)

        non_null = cleaned.dropna()
        non_null_count = len(non_null)

        if non_null_count == 0:
            result[col] = {
                "current_dtype": str(series.dtype),
                "non_null_count": 0,
                "numeric_convertible_count": 0,
                "numeric_ratio": 0.0,
                "numeric_examples": [],
                "numeric_non_convertible_examples": [],
                "datetime_convertible_count": 0,
                "datetime_ratio": 0.0,
                "detected_datetime_formats": {},
                "datetime_non_convertible_examples": [],
                "suggested_type": "undetermined",
            }
            continue

        # --- Détection numérique ---
        numeric_converted = pd.to_numeric(non_null, errors="coerce")
        numeric_convertible_count = int(numeric_converted.notna().sum())
        numeric_ratio = round(numeric_convertible_count / non_null_count, 2)

        numeric_examples = non_null[numeric_converted.notna()].head(5).tolist()
        numeric_non_convertible_examples = non_null[numeric_converted.isna()].head(5).tolist()

        # --- Détection datetime via _detect_datetime_formats (strict, formats explicites) ---
        dt_detection = _detect_datetime_formats(non_null)
        datetime_convertible_count = dt_detection["datetime_convertible_count"]
        datetime_ratio = dt_detection["datetime_ratio"]
        detected_datetime_formats = dt_detection["detected_formats"]
        datetime_non_convertible_examples = dt_detection["non_convertible_examples"]

        n_formats_detected = len(detected_datetime_formats)

        # --- Détermination du suggested_type ---
        numeric_certain = numeric_ratio >= threshold
        numeric_possible = possible_threshold <= numeric_ratio < threshold
        datetime_certain = datetime_ratio >= threshold
        datetime_possible = possible_threshold <= datetime_ratio < threshold

        if numeric_certain and datetime_certain:
            suggested_type = "ambiguous"
        elif numeric_certain:
            suggested_type = "numeric"
        elif numeric_possible and not datetime_certain and not datetime_possible:
            suggested_type = "possible_numeric"
        elif datetime_certain and n_formats_detected > 1:
            suggested_type = "datetime_mixed_formats"
        elif datetime_certain:
            suggested_type = "datetime"
        elif datetime_possible and n_formats_detected > 1:
            suggested_type = "possible_datetime_mixed_formats"
        elif datetime_possible:
            suggested_type = "possible_datetime"
        else:
            suggested_type = "keep_as_text"

        result[col] = {
            "current_dtype": str(series.dtype),
            "non_null_count": non_null_count,
            "numeric_convertible_count": numeric_convertible_count,
            "numeric_ratio": numeric_ratio,
            "numeric_examples": numeric_examples,
            "numeric_non_convertible_examples": numeric_non_convertible_examples,
            "datetime_convertible_count": datetime_convertible_count,
            "datetime_ratio": datetime_ratio,
            "detected_datetime_formats": detected_datetime_formats,
            "datetime_non_convertible_examples": datetime_non_convertible_examples,
            "suggested_type": suggested_type,
        }

    return result


def compute_numeric_profile(df):
    """
    Calcule les statistiques descriptives pour les colonnes numeriques.

    Pour chaque colonne int/float :
    - count, mean, std
    - min, Q1, median, Q3, max
    - mode(s) avec leur frequence
    - nombre et pourcentage de valeurs manquantes
    """
    result = {}

    for col in df.columns:
        series = df[col]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        non_null = series.dropna()
        non_null_count = len(non_null)
        null_count = int(series.isna().sum())
        n_rows = len(series)
        null_pct = round((null_count / n_rows) * 100, 2) if n_rows > 0 else 0.0

        if non_null_count == 0:
            result[col] = {
                "dtype": str(series.dtype),
                "non_null_count": 0,
                "null_count": null_count,
                "null_pct": null_pct,
                "mean": None,
                "std": None,
                "min": None,
                "Q1": None,
                "median": None,
                "Q3": None,
                "max": None,
                "mode": [],
                "mode_count": 0,
            }
            continue

        q1  = round(float(non_null.quantile(0.25)), 4)
        med = round(float(non_null.quantile(0.50)), 4)
        q3  = round(float(non_null.quantile(0.75)), 4)

        value_counts = non_null.value_counts()
        max_count = int(value_counts.iloc[0])
        modes = value_counts[value_counts == max_count].index.tolist()

        result[col] = {
            "dtype": str(series.dtype),
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_pct": null_pct,
            "mean": round(float(non_null.mean()), 4),
            "std": round(float(non_null.std()), 4),
            "min": round(float(non_null.min()), 4),
            "Q1": q1,
            "median": med,
            "Q3": q3,
            "max": round(float(non_null.max()), 4),
            "mode": modes,
            "mode_count": max_count,
        }

    return result

def compute_skewness(df):
    """
    Calcule le skewness de chaque colonne numérique et propose
    une transformation si nécessaire.

    Seuils :
    - |skew| < 0.5  -> "symmetric"
    - 0.5 <= |skew| < 1 -> "moderate"  (skew positif ou négatif)
    - |skew| >= 1   -> "high"          (skew positif ou négatif)

    Transformations suggérées (pour skew positif fort uniquement,
    car c'est le cas le plus fréquent en pratique) :
    - high positif  -> log1p (si min >= 0) ou yeo-johnson
    - moderate positif -> sqrt (si min >= 0) ou yeo-johnson
    - high/moderate négatif -> reflect + log1p, ou yeo-johnson
    - symmetric -> aucune
    """
    result = {}

    for col in df.columns:
        series = df[col]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        non_null = series.dropna()

        if len(non_null) < 3:
            result[col] = {
                "skewness": None,
                "level": "undetermined",
                "suggested_transform": None,
            }
            continue

        skew = round(float(non_null.skew()), 4)
        abs_skew = abs(skew)

        if abs_skew < 0.5:
            level = "symmetric"
            suggested_transform = None
        elif abs_skew < 1.0:
            level = "moderate"
        else:
            level = "high"

        if level in ("moderate", "high"):
            if skew > 0:
                # Queue à droite : valeurs extrêmes hautes
                if float(non_null.min()) >= 0:
                    suggested_transform = "sqrt" if level == "moderate" else "log1p"
                else:
                    suggested_transform = "yeo-johnson"
            else:
                # Queue à gauche : valeurs extrêmes basses
                suggested_transform = "reflect + log1p" if float(non_null.min()) >= 0 else "yeo-johnson"

        result[col] = {
            "skewness": skew,
            "level": level,
            "suggested_transform": suggested_transform,
        }

    return result


def compute_outliers(df, skewness_result=None, iqr_factor=1.5, mad_threshold=2.5, sample_size=5):
    """
    Détecte les outliers sur les colonnes numériques.

    Choix de la méthode par colonne selon le skewness :
    - symmetric / moderate -> IQR  (1.5 × IQR au-delà de Q1/Q3)
    - high / undetermined  -> MAD  (seuil de Iglewicz & Hoaglin, défaut 2.5)

    Si skewness_result n'est pas fourni, IQR est utilisé partout.

    Pour chaque colonne retourne :
    - method         : "IQR" ou "MAD"
    - outlier_count  : nombre de valeurs détectées
    - outlier_pct    : pourcentage sur les non-nulls
    - lower_bound    : borne basse calculée
    - upper_bound    : borne haute calculée
    - low_examples   : jusqu'à sample_size valeurs les plus basses détectées
    - high_examples  : jusqu'à sample_size valeurs les plus hautes détectées
    """
    result = {}

    for col in df.columns:
        series = df[col]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        non_null = series.dropna()
        non_null_count = len(non_null)

        if non_null_count < 4:
            result[col] = {
                "method": None,
                "outlier_count": 0,
                "outlier_pct": 0.0,
                "lower_bound": None,
                "upper_bound": None,
                "low_examples": [],
                "high_examples": [],
            }
            continue

        # Choix de la méthode selon le skewness de la colonne
        use_mad = False
        if skewness_result and col in skewness_result:
            level = skewness_result[col].get("level")
            if level in ("high", "undetermined"):
                use_mad = True

        if use_mad:
            median = float(non_null.median())
            mad = float((non_null - median).abs().median())

            # Evite division par zéro si toutes les valeurs sont identiques
            if mad == 0:
                mad = float((non_null - median).abs().mean())

            if mad == 0:
                result[col] = {
                    "method": "MAD",
                    "outlier_count": 0,
                    "outlier_pct": 0.0,
                    "lower_bound": median,
                    "upper_bound": median,
                    "low_examples": [],
                    "high_examples": [],
                }
                continue

            # Score MAD modifié : 0.6745 est le facteur de consistance
            # pour une distribution normale (Iglewicz & Hoaglin)
            modified_z = 0.6745 * (non_null - median).abs() / mad
            outlier_mask = modified_z > mad_threshold
            lower_bound = round(float(median - (mad_threshold / 0.6745) * mad), 4)
            upper_bound = round(float(median + (mad_threshold / 0.6745) * mad), 4)
            method = "MAD"

        else:
            q1 = float(non_null.quantile(0.25))
            q3 = float(non_null.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                result[col] = {
                    "method": "IQR",
                    "outlier_count": 0,
                    "outlier_pct": 0.0,
                    "lower_bound": round(q1, 4),
                    "upper_bound": round(q3, 4),
                    "low_examples": [],
                    "high_examples": [],
                }
                continue

            lower_bound = round(q1 - iqr_factor * iqr, 4)
            upper_bound = round(q3 + iqr_factor * iqr, 4)
            outlier_mask = (non_null < lower_bound) | (non_null > upper_bound)
            method = "IQR"

        outliers = non_null[outlier_mask]
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / non_null_count) * 100, 2)

        low_examples  = sorted(outliers[outliers < lower_bound].nsmallest(sample_size).tolist())
        high_examples = sorted(outliers[outliers > upper_bound].nlargest(sample_size).tolist(), reverse=True)

        result[col] = {
            "method": method,
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "low_examples": low_examples,
            "high_examples": high_examples,
        }

    return result


def compute_nan_analysis(df, skewness_result=None, drop_threshold=0.30):
    """
    Analyse les valeurs manquantes et propose une stratégie de gestion
    par colonne, en tenant compte de la distribution.

    Stratégies proposées :
    - "drop_column"    : taux de NaN > drop_threshold
    - "impute_mean"    : numérique, symétrique (skew faible)
    - "impute_median"  : numérique, skewé ou skewness inconnu
    - "impute_mode"    : catégorielle
    - "no_action"      : aucun NaN détecté

    Mécanisme de missing (MCAR / MAR / MNAR) :
    Pour chaque colonne avec des NaN, on teste la corrélation entre le
    masque binaire de NaN (1=manquant, 0=présent) et chaque autre colonne
    numérique via le test point-biserial (scipy.stats.pointbiserialr).

    - MCAR (Missing Completely At Random) : aucune corrélation significative
      détectée. La donnée manque de façon purement aléatoire, indépendamment
      de toute autre variable. Les imputations simples (moyenne, médiane) sont
      appropriées.

    - MAR (Missing At Random) : le fait qu'une valeur soit manquante est
      corrélé à d'autres variables observables (ex: les NaN dans 'Salary'
      sont concentrés chez les jeunes). Mal nommé historiquement, MAR signifie
      en réalité "explicable par d'autres variables". Une imputation
      conditionnelle (KNN, régression) est recommandée pour éviter un biais.

    - MNAR (Missing Not At Random) : aucune variable observée n'explique le
      pattern, mais ce n'est pas aléatoire non plus (ex: les gens très riches
      ne déclarent pas leur salaire). Non-prouvable algorithmiquement — signalé
      comme hypothèse quand aucune corrélation MAR n'est trouvée mais que le
      taux de NaN est élevé (> 10%). Toute imputation est risquée dans ce cas.

    Parameters
    ----------
    df : pd.DataFrame
    skewness_result : dict | None
        Résultat de compute_skewness(), utilisé pour choisir mean vs median.
    drop_threshold : float
        Taux de NaN au-delà duquel on propose la suppression (défaut: 0.30).
    """
    from scipy.stats import pointbiserialr

    result = {}
    n_rows = len(df)

    # Colonnes numériques disponibles pour le test de corrélation
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / n_rows) * 100, 2) if n_rows > 0 else 0.0

        if null_count == 0:
            result[col] = {
                "null_count": 0,
                "null_pct": 0.0,
                "proposed_action": "no_action",
                "imputation_method": None,
                "missing_mechanism": None,
                "correlated_with": [],
            }
            continue

        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_categorical = series.dtype == "object" or pd.api.types.is_string_dtype(series)

        # --- Stratégie d'imputation ---
        if null_pct / 100 > drop_threshold:
            proposed_action = "drop_column"
            imputation_method = None
        elif is_categorical:
            proposed_action = "impute"
            imputation_method = "mode"
        elif is_numeric:
            skew_level = None
            if skewness_result and col in skewness_result:
                skew_level = skewness_result[col].get("level")

            if skew_level in ("high", "moderate"):
                imputation_method = "median"
            else:
                # symmetric ou inconnu -> mean
                imputation_method = "mean"
            proposed_action = "impute"
        else:
            proposed_action = "impute"
            imputation_method = "mode"

        # --- Détection du mécanisme (MCAR / MAR / MNAR) ---
        missing_mechanism = None
        correlated_with = []

        nan_mask = series.isna().astype(int)
        # Le test n'a de sens que si le masque a de la variance
        if nan_mask.std() > 0:
            for other_col in numeric_cols:
                if other_col == col:
                    continue
                other_series = df[other_col].dropna()
                # Aligner les index
                common_idx = nan_mask.index.intersection(other_series.index)
                if len(common_idx) < 10:
                    continue
                mask_aligned  = nan_mask.loc[common_idx]
                other_aligned = other_series.loc[common_idx]
                if other_aligned.std() == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r, p = pointbiserialr(mask_aligned, other_aligned)
                    if pd.isna(r) or pd.isna(p):
                        continue
                    if p < 0.05 and abs(r) > 0.1:
                        correlated_with.append({
                            "column": other_col,
                            "r": round(float(r), 3),
                            "p_value": round(float(p), 4),
                        })
                except Exception:
                    continue

        if correlated_with:
            missing_mechanism = "MAR"
            # Si MAR, l'imputation simple est sous-optimale — on surcharge
            if proposed_action == "impute":
                imputation_method = "knn_or_regression"
        elif null_pct > 10:
            missing_mechanism = "MNAR"
        else:
            missing_mechanism = "MCAR"

        result[col] = {
            "null_count": null_count,
            "null_pct": null_pct,
            "proposed_action": proposed_action,
            "imputation_method": imputation_method,
            "missing_mechanism": missing_mechanism,
            "correlated_with": correlated_with,
        }

    return result


def compute_correlations(df, high_threshold=0.85, show_full_matrix_max_cols=10,
                         cramers_v_threshold=0.25, target_col=None):
    """
    Calcule la matrice de corrélation de Pearson sur les colonnes numériques
    et les associations Chi² / Cramér's V sur les colonnes catégorielles.

    Parameters
    ----------
    df : pd.DataFrame
    high_threshold : float
        Seuil Pearson |r| au-delà duquel une paire numérique est signalée (défaut: 0.85).
    show_full_matrix_max_cols : int
        Si le nombre de colonnes numériques est <= cette valeur,
        la matrice complète est incluse dans le résultat (défaut: 10).
    cramers_v_threshold : float
        Seuil Cramér's V au-delà duquel une association catégorielle est signalée.
        Seuils de référence : 0.1 faible, 0.25 modérée, 0.35 forte (défaut: 0.25).
        Seules les paires avec p < 0.05 ET V >= seuil sont rapportées.

    Retourne
    --------
    dict avec :
    - "high_pairs"         : paires numériques avec |r| >= threshold
    - "matrix"             : matrice Pearson complète ou None
    - "numeric_cols"       : colonnes numériques analysées
    - "categorical_associations" : paires catégorielles avec V >= threshold et p < 0.05
    - "categorical_cols"   : colonnes catégorielles analysées
    """
    exclude = {target_col} if target_col else set()
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude
    ]

    high_pairs = []
    matrix = None

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(method="pearson")

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                r = corr_matrix.loc[col_a, col_b]
                if pd.isna(r):
                    continue
                if abs(r) >= high_threshold:
                    high_pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "r": round(float(r), 4),
                    })

        high_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

        if len(numeric_cols) <= show_full_matrix_max_cols:
            matrix = {
                col: {
                    other: round(float(corr_matrix.loc[col, other]), 4)
                    for other in numeric_cols
                }
                for col in numeric_cols
            }

    # --- Associations catégorielles (Chi² + Cramér's V) ---
    from scipy.stats import chi2_contingency
    import warnings as _warnings

    # Colonnes catégorielles : string/object + numériques à faible cardinalité
    # (low_cardinality_max_unique = 10 par défaut, cohérent avec feature_quality)
    # On exclut les colonnes à trop haute cardinalité (> 50% de valeurs uniques)
    # car le Chi² y produit des V artificiellement élevés non interprétables.
    n_rows_total = len(df)
    cat_cols = [
        c for c in df.columns
        if (
            df[c].dtype == "object"
            or pd.api.types.is_string_dtype(df[c])
            or (pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 10)
        )
        and (df[c].nunique(dropna=True) / max(df[c].count(), 1)) <= 0.5
        and c not in exclude
    ]

    categorical_associations = []

    for i, col_a in enumerate(cat_cols):
        for col_b in cat_cols[i + 1:]:
            valid = df[[col_a, col_b]].dropna()
            n = len(valid)
            if n < 10:
                continue

            try:
                contingency = pd.crosstab(valid[col_a], valid[col_b])
                r, c = contingency.shape
                if r < 2 or c < 2:
                    continue

                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    chi2, p, dof, _ = chi2_contingency(contingency)

                if pd.isna(chi2) or pd.isna(p):
                    continue

                # Cramér's V corrigé (Bergsma 2013) — corrige le biais dû
                # à la haute cardinalité et aux petits échantillons.
                phi2 = chi2 / n
                phi2_corr = max(0.0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
                r_corr = r - (r - 1) ** 2 / (n - 1)
                c_corr = c - (c - 1) ** 2 / (n - 1)
                denom = min(r_corr - 1, c_corr - 1)
                if denom <= 0:
                    continue
                v = round(float((phi2_corr / denom) ** 0.5), 4)

                if p < 0.05 and v >= cramers_v_threshold:
                    strength = "strong" if v >= 0.35 else "moderate"
                    categorical_associations.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "cramers_v": v,
                        "p_value": round(float(p), 4),
                        "strength": strength,
                    })

            except Exception:
                continue

    categorical_associations.sort(key=lambda x: x["cramers_v"], reverse=True)

    return {
        "high_pairs": high_pairs,
        "matrix": matrix,
        "numeric_cols": numeric_cols,
        "categorical_associations": categorical_associations,
        "categorical_cols": cat_cols,
    }


def compute_row_analysis(df, drop_threshold=0.50):
    """
    Analyse les valeurs manquantes par ligne.

    Pour chaque ligne, calcule le ratio de NaN. Les lignes dépassant
    drop_threshold sont marquées pour suppression.

    Retourne
    --------
    dict avec :
    - "drop_threshold"    : seuil utilisé
    - "total_rows"        : nombre total de lignes
    - "rows_to_drop"      : nombre de lignes au-dessus du seuil
    - "rows_to_drop_pct"  : pourcentage correspondant
    - "rows_to_drop_idx"  : index des lignes concernées
    - "distribution"      : répartition en 4 buckets (0%, 1-25%, 25-50%, >50%)
    """
    n_cols = df.shape[1]
    if n_cols == 0:
        return {
            "drop_threshold": drop_threshold,
            "total_rows": len(df),
            "rows_to_drop": 0,
            "rows_to_drop_pct": 0.0,
            "rows_to_drop_idx": [],
            "distribution": {},
        }

    nan_ratio_per_row = df.isna().sum(axis=1) / n_cols

    drop_mask        = nan_ratio_per_row >= drop_threshold
    rows_to_drop     = int(drop_mask.sum())
    rows_to_drop_pct = round((rows_to_drop / len(df)) * 100, 2) if len(df) > 0 else 0.0
    rows_to_drop_idx = df.index[drop_mask].tolist()

    distribution = {
        "0%":      int((nan_ratio_per_row == 0).sum()),
        "1-25%":   int(((nan_ratio_per_row > 0) & (nan_ratio_per_row <= 0.25)).sum()),
        "25-50%":  int(((nan_ratio_per_row > 0.25) & (nan_ratio_per_row < drop_threshold)).sum()),
        f">={int(drop_threshold * 100)}%": rows_to_drop,
    }

    return {
        "drop_threshold": drop_threshold,
        "total_rows": len(df),
        "rows_to_drop": rows_to_drop,
        "rows_to_drop_pct": rows_to_drop_pct,
        "rows_to_drop_idx": rows_to_drop_idx,
        "distribution": distribution,
    }


def compute_feature_quality(
    df,
    quasi_constant_threshold=0.95,
    low_cardinality_max_unique=10,
    high_cardinality_id_ratio=0.95,
    feature_display=True,
):
    """
    Analyse la qualité des features pour le ML. Regroupe trois détections :

    1. Colonnes quasi-constantes
       Une colonne où une seule valeur représente >= quasi_constant_threshold
       des lignes non-nulles n'apporte quasiment aucune information à un modèle.
       Seuil défaut : 95%.

    2. Colonnes numériques à faible cardinalité
       Une colonne int/float avec <= low_cardinality_max_unique valeurs distinctes
       est probablement une variable catégorielle encodée (0/1, 1/2/3...).
       Seuil défaut : 10 valeurs uniques max.

    3. Colonnes potentiellement identifiantes
       Une colonne object avec un ratio d'unicité >= high_cardinality_id_ratio
       est probablement un identifiant, un nom propre ou une donnée personnelle.
       Elle ne devrait pas être utilisée comme feature.
       Seuil défaut : 95% de valeurs uniques.

    Retourne
    --------
    dict {col: info} pour chaque colonne avec au moins un signal détecté.
    info contient :
    - "issues"              : liste des problèmes détectés (peut en avoir plusieurs)
    - "quasi_constant"      : bool
    - "dominant_value"      : valeur dominante et son ratio (si quasi_constant)
    - "low_cardinality"     : bool
    - "n_unique"            : nombre de valeurs uniques (si low_cardinality)
    - "unique_values"       : liste des valeurs (si low_cardinality)
    - "potential_id"        : bool
    - "unique_ratio"        : ratio d'unicité (si potential_id)
    """
    result = {}
    n_rows = len(df)

    if n_rows == 0:
        return result

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        non_null_count = len(non_null)

        if non_null_count == 0:
            continue

        issues = []
        info = {
            "issues": issues,
            "quasi_constant": False,
            "dominant_value": None,
            "dominant_ratio": None,
            "low_cardinality": False,
            "n_unique": None,
            "unique_values": None,
            "potential_id": False,
            "unique_ratio": None,
        }

        # --- 1. Quasi-constante ---
        value_counts = non_null.value_counts(normalize=True)
        dominant_ratio = float(value_counts.iloc[0])
        if dominant_ratio >= quasi_constant_threshold:
            info["quasi_constant"] = True
            raw = value_counts.index[0]
            info["dominant_value"] = raw.item() if hasattr(raw, "item") else raw
            info["dominant_ratio"] = round(dominant_ratio, 4)
            issues.append("quasi_constant")

        # --- 2. Numérique à faible cardinalité ---
        if pd.api.types.is_numeric_dtype(series):
            n_unique = int(non_null.nunique())
            if n_unique <= low_cardinality_max_unique:
                info["low_cardinality"] = True
                info["n_unique"] = n_unique
                info["unique_values"] = sorted(non_null.unique().tolist())
                issues.append("low_cardinality")

        # potential_id is computed separately in quick_report
        # on the cleaned df (post-duplicates + post-row-filter)

        if issues:
            result[col] = info

    return result


def compute_vif(df, vif_threshold=10.0):
    """
    Calcule le Variance Inflation Factor (VIF) pour chaque colonne numérique continue.

    Le VIF mesure à quel point une variable est redondante par rapport aux autres :
    il régresse chaque colonne sur toutes les autres et calcule 1 / (1 - R²).

    Interprétation :
    - VIF = 1         : aucune corrélation avec les autres variables
    - 1 < VIF < 5    : corrélation faible à modérée, acceptable
    - 5 <= VIF < 10  : corrélation élevée, à surveiller
    - VIF >= 10      : multicolinéarité forte — la variable est quasi-redondante

    Seules les colonnes numériques continues sont analysées (les colonnes
    à faible cardinalité <= 10 valeurs uniques sont exclues car elles sont
    déjà couvertes par le Chi² / Cramér's V).

    Parameters
    ----------
    df : pd.DataFrame
    vif_threshold : float
        Seuil au-delà duquel une colonne est signalée (défaut: 10.0).
    """
    import warnings as _warnings

    # Colonnes numériques continues uniquement (exclure les quasi-catégorielles)
    continuous_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() > 10
    ]

    if len(continuous_cols) < 2:
        return {
            "vif_threshold": vif_threshold,
            "columns_analysed": continuous_cols,
            "results": {},
            "high_vif": [],
        }

    # Supprimer les lignes avec NaN sur ces colonnes pour le calcul
    df_clean = df[continuous_cols].dropna()

    if len(df_clean) < len(continuous_cols) + 1:
        return {
            "vif_threshold": vif_threshold,
            "columns_analysed": continuous_cols,
            "results": {},
            "high_vif": [],
        }

    results = {}
    high_vif = []
    import numpy as _np

    for col in continuous_cols:
        other_cols = [c for c in continuous_cols if c != col]
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                X = df_clean[other_cols].values
                y = df_clean[col].values
                # Ajouter une constante pour la régression
                X_const = _np.column_stack([_np.ones(len(X)), X])
                # Résolution par moindres carrés
                coeffs, _, _, _ = _np.linalg.lstsq(X_const, y, rcond=None)
                y_pred = X_const @ coeffs
                ss_res = ((y - y_pred) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum()
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                r2 = min(max(r2, 0.0), 0.9999)
                vif_val = round(1.0 / (1.0 - r2), 4)
        except Exception:
            vif_val = None

        if vif_val is None:
            results[col] = {"vif": None, "flag": "undetermined"}
            continue

        if vif_val >= vif_threshold:
            flag = "high"
            high_vif.append({"column": col, "vif": vif_val})
        elif vif_val >= 5:
            flag = "moderate"
        else:
            flag = "ok"

        results[col] = {"vif": vif_val, "flag": flag}

    high_vif.sort(key=lambda x: x["vif"], reverse=True)

    return {
        "vif_threshold": vif_threshold,
        "columns_analysed": continuous_cols,
        "results": results,
        "high_vif": high_vif,
    }


def compute_target_analysis(df, target_col, cramers_v_threshold=0.25):
    """
    Analyse ciblée sur la colonne à prédire.

    Détecte automatiquement si la target est catégorielle ou numérique et calcule :

    Pour les deux types :
    - leakage_candidates : colonnes avec corrélation quasi-parfaite (r > 0.95 ou V > 0.95)

    Pour les targets catégorielles (classification) :
    - class_balance : répartition des classes et ratio minoritaire
    - imbalanced : bool, True si la classe minoritaire < 20%
    - feature_correlations : V de Cramér ou point-biserial par feature, triées desc

    Pour les targets numériques (régression) :
    - skewness : skew de la target
    - suggested_transform : transformation recommandée
    - feature_correlations : Pearson par feature numérique, triées desc
    - outlier_count / outlier_pct : outliers dans la target
    """
    import warnings as _warnings
    from scipy.stats import pointbiserialr

    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not found in DataFrame."}

    target = df[target_col].dropna()
    n = len(target)
    result = {"target_col": target_col, "n": n}

    is_categorical = (
        df[target_col].dtype == "object"
        or pd.api.types.is_string_dtype(df[target_col])
        or df[target_col].nunique() <= 10
    )

    # --- Classification ---
    if is_categorical:
        result["task_type"] = "classification"
        vc = target.value_counts()
        total = vc.sum()
        class_balance = {str(k): {"count": int(v), "pct": round(v/total*100, 2)}
                         for k, v in vc.items()}
        minority_pct = round(vc.iloc[-1] / total * 100, 2) if len(vc) > 1 else 100.0
        result["class_balance"] = class_balance
        result["minority_pct"] = minority_pct
        result["n_classes"] = len(vc)
        result["imbalanced"] = minority_pct < 20.0

        # Corrélations features -> target
        feature_correlations = []
        cat_cols = [c for c in df.columns if c != target_col
                    and (df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])
                         or df[c].nunique() <= 10)]
        num_cols = [c for c in df.columns if c != target_col
                    and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 10]

        for col in cat_cols:
            valid = df[[col, target_col]].dropna()
            if len(valid) < 10:
                continue
            try:
                contingency = pd.crosstab(valid[col], valid[target_col])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                from scipy.stats import chi2_contingency
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    chi2, p, _, _ = chi2_contingency(contingency)
                r_val, c_val = contingency.shape
                phi2_corr = max(0, chi2/len(valid) - (r_val-1)*(c_val-1)/(len(valid)-1))
                r_corr = r_val - (r_val-1)**2/(len(valid)-1)
                c_corr = c_val - (c_val-1)**2/(len(valid)-1)
                denom = min(r_corr-1, c_corr-1)
                if denom > 0:
                    v = round(float((phi2_corr/denom)**0.5), 4)
                    feature_correlations.append({"column": col, "metric": "cramers_v", "value": v, "p_value": round(float(p), 4)})
            except Exception:
                continue

        for col in num_cols:
            valid = df[[col, target_col]].dropna()
            if len(valid) < 10:
                continue
            try:
                mask = valid[target_col].astype(str)
                # encode target as binary if 2 classes, else skip
                classes = mask.unique()
                if len(classes) == 2:
                    binary = (mask == classes[0]).astype(int)
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")
                        r, p = pointbiserialr(binary, valid[col])
                    if not pd.isna(r):
                        feature_correlations.append({"column": col, "metric": "point_biserial_r", "value": round(float(abs(r)), 4), "p_value": round(float(p), 4)})
            except Exception:
                continue

        feature_correlations.sort(key=lambda x: x["value"], reverse=True)
        result["feature_correlations"] = feature_correlations

    # --- Régression ---
    else:
        result["task_type"] = "regression"
        skew_val = round(float(target.skew()), 4)
        result["skewness"] = skew_val
        abs_skew = abs(skew_val)
        if abs_skew < 0.5:
            result["suggested_transform"] = None
        elif skew_val > 0:
            result["suggested_transform"] = "log1p" if float(target.min()) >= 0 else "yeo-johnson"
        else:
            result["suggested_transform"] = "reflect + log1p" if float(target.min()) >= 0 else "yeo-johnson"

        # Outliers dans la target
        q1, q3 = float(target.quantile(0.25)), float(target.quantile(0.75))
        iqr = q3 - q1
        outlier_mask = (target < q1 - 1.5*iqr) | (target > q3 + 1.5*iqr)
        result["outlier_count"] = int(outlier_mask.sum())
        result["outlier_pct"] = round(result["outlier_count"] / n * 100, 2)

        # Corrélations Pearson features numériques -> target
        # Pour la régression : toutes les colonnes numériques, y compris low-cardinality
        num_cols = [c for c in df.columns if c != target_col
                    and pd.api.types.is_numeric_dtype(df[c])]
        feature_correlations = []
        for col in num_cols:
            valid = df[[col, target_col]].dropna()
            if len(valid) < 10:
                continue
            try:
                r = round(float(valid[col].corr(valid[target_col])), 4)
                if not pd.isna(r):
                    feature_correlations.append({"column": col, "metric": "pearson_r", "value": round(abs(r), 4), "signed_r": r})
            except Exception:
                continue
        feature_correlations.sort(key=lambda x: x["value"], reverse=True)
        result["feature_correlations"] = feature_correlations

    # --- Leakage candidates (commun) ---
    leakage = []
    for entry in result.get("feature_correlations", []):
        if entry["value"] >= 0.95:
            leakage.append(entry["column"])
    result["leakage_candidates"] = leakage

    return result


def quick_report(
    df,
    overview=True,
    visible_missing=True,
    categorical_profile=True,
    duplicates=True,
    type_issues=True,
    numeric_profile=True,
    skewness=True,
    outliers=True,
    nan_analysis=True,
    nan_drop_threshold=0.30,
    correlations=True,
    correlation_threshold=0.85,
    cramers_v_threshold=0.25,
    max_correlation_warnings=10,
    row_analysis=True,
    row_drop_threshold=0.50,
    apply_row_filter=True,
    feature_quality=True,
    quasi_constant_threshold=0.95,
    low_cardinality_max_unique=10,
    high_cardinality_id_ratio=0.95,
    feature_display=True,
    vif=True,
    vif_threshold=10.0,
    target=None,
):
    """
    Fonction principale qui orchestre les steps activées.
    """
    config = {
        "overview": overview,
        "visible_missing": visible_missing,
        "categorical_profile": categorical_profile,
        "duplicates": duplicates,
        "type_issues": type_issues,
        "numeric_profile": numeric_profile,
        "skewness": skewness,
        "outliers": outliers,
        "nan_analysis": nan_analysis,
        "nan_drop_threshold": nan_drop_threshold,
        "correlations": correlations,
        "correlation_threshold": correlation_threshold,
        "cramers_v_threshold": cramers_v_threshold,
        "max_correlation_warnings": max_correlation_warnings,
        "row_analysis": row_analysis,
        "row_drop_threshold": row_drop_threshold,
        "apply_row_filter": apply_row_filter,
        "feature_quality": feature_quality,
        "quasi_constant_threshold": quasi_constant_threshold,
        "low_cardinality_max_unique": low_cardinality_max_unique,
        "high_cardinality_id_ratio": high_cardinality_id_ratio,
        "feature_display": feature_display,
        "vif": vif,
        "vif_threshold": vif_threshold,
        "target": target,
    }

    report = Report(df, config)

    if config["overview"]:
        report.add_result("overview", compute_overview(df))

    if config["feature_quality"]:
        result = compute_feature_quality(
            df,
            quasi_constant_threshold=config["quasi_constant_threshold"],
            low_cardinality_max_unique=config["low_cardinality_max_unique"],
            high_cardinality_id_ratio=config["high_cardinality_id_ratio"],
        )
        report.add_result("feature_quality", result)

        for col, info in result.items():
            for issue in info["issues"]:
                if issue == "quasi_constant":
                    report.add_warning(
                        f"Column '{col}' is quasi-constant: '{info['dominant_value']}' "
                        f"represents {info['dominant_ratio']*100:.1f}% of non-null values. "
                        f"Low predictive value for ML."
                    )
                elif issue == "low_cardinality":
                    report.add_warning(
                        f"Column '{col}' is numeric but has only {info['n_unique']} distinct values "
                        f"{info['unique_values']}. Consider treating as categorical."
                    )
                elif issue == "potential_id":
                    pass  # handled post-cleaning below

    if config["visible_missing"]:
        result = compute_visible_missing(df)
        report.add_result("visible_missing", result)

        for col, pct in result["missing_pct_by_column"].items():
            if pct > 30:
                report.add_warning(
                    f"Column '{col}' has {pct:.2f}% visible missing values"
                )

    if config["duplicates"]:
        result = compute_duplicates(df)
        report.add_result("duplicates", result)

        if result["duplicate_count"] > 0:
            report.add_warning(
                f"{result['duplicate_count']} duplicated rows detected ({result['duplicate_pct']:.2f}%)"
            )
    if config["categorical_profile"]:
        result = compute_categorical_profile(df)
        report.add_result("categorical_profile", result)

        for col, col_info in result.items():
            if col_info["hidden_missing_count"] > 0:
                report.add_warning(
                    f"Column '{col}' contains {col_info['hidden_missing_count']} hidden missing values"
                )

            if col_info["case_collisions"]:
                report.add_warning(
                    f"Column '{col}' contains values that differ only by letter case"
                )
    
    if config["type_issues"]:
        result = compute_type_issues(df)
        report.add_result("type_issues", result)

        for col, info in result.items():
            suggested = info["suggested_type"]

            if suggested == "numeric":
                report.add_warning(
                    f"Column '{col}' is typed as {info['current_dtype']} but looks numeric "
                    f"({info['numeric_ratio']:.2f} convertible)"
                )

            elif suggested == "possible_numeric":
                report.add_warning(
                    f"Column '{col}' may be numeric "
                    f"({info['numeric_ratio']:.2f} convertible). Check examples: "
                    f"{info['numeric_examples']}"
                )

            elif suggested == "datetime":
                report.add_warning(
                    f"Column '{col}' is typed as {info['current_dtype']} but looks datetime "
                    f"({info['datetime_ratio']:.2f} convertible)"
                )

            elif suggested == "datetime_mixed_formats":
                format_examples = {
                    fmt: meta["example"]
                    for fmt, meta in info["detected_datetime_formats"].items()
                }
                report.add_warning(
                    f"Column '{col}' looks datetime but contains multiple formats "
                    f"({info['datetime_ratio']:.2f} convertible). Examples by format: "
                    f"{format_examples}"
                )

            elif suggested == "possible_datetime":
                format_examples = {
                    fmt: meta["example"]
                    for fmt, meta in info["detected_datetime_formats"].items()
                }
                report.add_warning(
                    f"Column '{col}' may be datetime "
                    f"({info['datetime_ratio']:.2f} convertible). Detected formats: "
                    f"{format_examples}"
                )

            elif suggested == "possible_datetime_mixed_formats":
                format_examples = {
                    fmt: meta["example"]
                    for fmt, meta in info["detected_datetime_formats"].items()
                }
                report.add_warning(
                    f"Column '{col}' may contain mixed datetime formats "
                    f"({info['datetime_ratio']:.2f} convertible). Examples by format: "
                    f"{format_examples}"
                )

            elif suggested == "ambiguous":
                report.add_warning(
                    f"Column '{col}' is ambiguous: looks both numeric "
                    f"({info['numeric_ratio']:.2f}) and datetime "
                    f"({info['datetime_ratio']:.2f}). Manual check required."
                )
    

    if config["numeric_profile"]:
        report.add_result("numeric_profile", compute_numeric_profile(df))

    if config["skewness"]:
        result = compute_skewness(df)
        report.add_result("skewness", result)

        for col, info in result.items():
            level = info["level"]
            skew = info["skewness"]
            transform = info["suggested_transform"]

            if level == "high":
                report.add_warning(
                    f"Column '{col}' has high skewness ({skew}). "
                    f"Suggested transform: {transform}"
                )
            elif level == "moderate":
                report.add_warning(
                    f"Column '{col}' has moderate skewness ({skew}). "
                    f"Consider: {transform}"
                )

    # --- Row analysis + filtered df for downstream steps ---
    df_analysis = df  # par défaut, toutes les lignes

    if config["row_analysis"]:
        row_result = compute_row_analysis(df, drop_threshold=config["row_drop_threshold"])
        report.add_result("row_analysis", row_result)

        rows_dropped   = row_result["rows_to_drop"]
        rows_total     = row_result["total_rows"]
        threshold_pct  = int(config["row_drop_threshold"] * 100)
        dropped_pct    = row_result["rows_to_drop_pct"]

        if rows_dropped > 0:
            report.add_warning(
                f"Row analysis: {rows_dropped}/{rows_total} rows ({dropped_pct}%) have "
                f">={threshold_pct}% missing values and are flagged for removal. "
                f"NOTE: the following steps (nan_analysis, outliers, correlations) "
                f"are computed on the remaining {rows_total - rows_dropped} rows only."
            )
            if config["apply_row_filter"]:
                keep_idx   = df.index.difference(row_result["rows_to_drop_idx"])
                df_analysis = df.loc[keep_idx]

    # --- Potential ID detection on cleaned df (post-duplicates + post-row-filter) ---
    if config["feature_quality"]:
        fq = report.get("feature_quality") or {}
        df_id_check = df_analysis.drop_duplicates()
        for col in df_id_check.columns:
            non_null = df_id_check[col].dropna()
            non_null_count = len(non_null)
            if non_null_count == 0:
                continue
            unique_ratio = round(non_null.nunique() / non_null_count, 4)
            if unique_ratio >= 1.0:
                if col in fq:
                    if "potential_id" not in fq[col]["issues"]:
                        fq[col]["potential_id"] = True
                        fq[col]["unique_ratio"] = unique_ratio
                        fq[col]["issues"].append("potential_id")
                else:
                    fq[col] = {
                        "issues": ["potential_id"],
                        "quasi_constant": False,
                        "dominant_value": None,
                        "dominant_ratio": None,
                        "low_cardinality": False,
                        "n_unique": None,
                        "unique_values": None,
                        "potential_id": True,
                        "unique_ratio": unique_ratio,
                    }
                report.add_warning(
                    f"Column '{col}' has 100% unique values after cleaning "
                    f"(post-duplicates + post-row-filter) — suspected identifier. "
                    f"Should not be used as a feature."
                )
        # Update the stored result
        report.results["feature_quality"] = fq

    if config["outliers"]:
        skewness_result = report.get("skewness")
        result = compute_outliers(df_analysis, skewness_result=skewness_result)
        report.add_result("outliers", result)

        for col, info in result.items():
            if info["outlier_count"] > 0:
                report.add_warning(
                    f"Column '{col}' has {info['outlier_count']} outliers "
                    f"({info['outlier_pct']}%) detected via {info['method']}. "
                    f"Bounds: [{info['lower_bound']}, {info['upper_bound']}]"
                )

    if config["nan_analysis"]:
        skewness_result = report.get("skewness")
        result = compute_nan_analysis(
            df_analysis,
            skewness_result=skewness_result,
            drop_threshold=config["nan_drop_threshold"],
        )
        report.add_result("nan_analysis", result)

        MAR_WARNING = (
            "MAR (Missing At Random): missingness is correlated with other observed variables. "
            "Simple imputation (mean/median) may introduce bias. "
            "Consider conditional imputation (KNN, regression)."
        )
        MNAR_WARNING = (
            "MNAR (Missing Not At Random): no observed variable explains the missing pattern, "
            "but the rate is high enough to suspect a systematic cause "
            "(e.g. respondents omitting sensitive data). "
            "Any imputation carries risk — flag this column for domain review."
        )

        for col, info in result.items():
            if info["proposed_action"] == "no_action":
                continue

            mechanism = info["missing_mechanism"]
            action    = info["proposed_action"]
            method    = info["imputation_method"]
            pct       = info["null_pct"]
            corr      = info["correlated_with"]

            if action == "drop_column":
                semantic_note = ""
                if pct > 70:
                    semantic_note = (
                        " NOTE: Very high missing rate — verify whether NaN encodes "
                        "absence of feature (e.g. no garage, no pool) rather than truly missing data. "
                        "If so, consider replacing NaN with a 'None' category instead of dropping."
                    )
                report.add_warning(
                    f"Column '{col}' is {pct}% missing (threshold: {config['nan_drop_threshold']*100:.0f}%). "
                    f"Proposed action: drop column.{semantic_note}"
                )
            else:
                report.add_warning(
                    f"Column '{col}' has {pct}% missing values. "
                    f"Proposed imputation: {method}. "
                    f"Missing mechanism: {mechanism}."
                )

            if mechanism == "MAR":
                corr_summary = ", ".join(
                    f"{c['column']} (r={c['r']}, p={c['p_value']})"
                    for c in corr
                )
                report.add_warning(
                    f"  -> {MAR_WARNING} "
                    f"Correlated with: {corr_summary}."
                )
            elif mechanism == "MNAR":
                report.add_warning(f"  -> {MNAR_WARNING}")

    if config["correlations"]:
        result = compute_correlations(df_analysis, high_threshold=config["correlation_threshold"], cramers_v_threshold=config["cramers_v_threshold"], target_col=config["target"])
        report.add_result("correlations", result)

        for pair in result["high_pairs"]:
            report.add_warning(
                f"High correlation between '{pair['col_a']}' and '{pair['col_b']}': "
                f"r={pair['r']}. This may indicate multicollinearity — "
                f"consider dropping or combining one of these features before training a linear model."
            )

        cat_assocs = result.get("categorical_associations", [])
        total_cat_assocs = len(cat_assocs)
        max_w = config["max_correlation_warnings"]

        if total_cat_assocs > max_w:
            report.add_warning(
                f"Categorical associations: {total_cat_assocs} significant pairs detected "
                f"(showing top {max_w} by Cramér\'s V). "
                f"Full list accessible via report.get(\"correlations\")[\"categorical_associations\"]."
            )

        seen_pairs = set()
        shown = 0
        for assoc in cat_assocs:
            if shown >= max_w:
                break
            pair_key = frozenset([assoc["col_a"], assoc["col_b"]])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            v = assoc["cramers_v"]
            strength = assoc["strength"]

            if v >= 0.9:
                redundancy_note = (
                    "These columns carry almost identical information — "
                    "keeping both is likely redundant. Consider dropping one."
                )
            elif v >= 0.35:
                redundancy_note = (
                    "These columns share significant information overlap — "
                    "one may partly explain the other. Consider this before encoding or feature selection."
                )
            else:
                redundancy_note = (
                    "These columns show a moderate association — "
                    "they are not fully independent. Keep in mind for feature selection."
                )

            report.add_warning(
                f"Categorical association ({strength}): '{assoc['col_a']}' <-> '{assoc['col_b']}' "
                f"Cramér\'s V={v} (p={assoc['p_value']}). {redundancy_note}"
            )
            shown += 1

    if config["target"] is not None:
        target_result = compute_target_analysis(
            df,
            target_col=config["target"],
            cramers_v_threshold=config["cramers_v_threshold"],
        )
        report.add_result("target_analysis", target_result)

        task = target_result.get("task_type")
        tgt  = config["target"]

        if task == "classification":
            if target_result.get("imbalanced"):
                minority = target_result["minority_pct"]
                n_cls    = target_result["n_classes"]
                report.add_warning(
                    f"Target '{tgt}': class imbalance detected — minority class represents "
                    f"{minority}% of data ({n_cls} classes total). "
                    f"Consider SMOTE, undersampling, or class_weight adjustments before training."
                )
            else:
                minority = target_result["minority_pct"]
                report.add_warning(
                    f"Target '{tgt}': class distribution OK — minority class at {minority}%."
                )

        elif task == "regression":
            skew = target_result.get("skewness", 0)
            transform = target_result.get("suggested_transform")
            if abs(skew) >= 0.5:
                report.add_warning(
                    f"Target '{tgt}': skewness={skew}. "
                    f"Recommended transform before modeling: {transform}."
                )
            oc = target_result.get("outlier_count", 0)
            op = target_result.get("outlier_pct", 0)
            if oc > 0:
                report.add_warning(
                    f"Target '{tgt}': {oc} outliers ({op}%) detected via IQR. "
                    f"These will disproportionately affect regression loss — consider capping or transforming."
                )

        for col in target_result.get("leakage_candidates", []):
            report.add_warning(
                f"LEAKAGE RISK: '{col}' has near-perfect correlation with target '{tgt}'. "
                f"Verify this is not a derived or post-event feature."
            )

    if config["vif"]:
        vif_cols = [c for c in df_analysis.columns if c != config["target"]]
        result = compute_vif(df_analysis[vif_cols], vif_threshold=config["vif_threshold"])
        report.add_result("vif", result)

        for entry in result["high_vif"]:
            report.add_warning(
                f"High VIF for '{entry['column']}': {entry['vif']} "
                f"(threshold: {config['vif_threshold']}). "
                f"This column is largely explained by other numeric variables — "
                f"it carries redundant information and may harm linear models. "
                f"Consider removing it or applying dimensionality reduction."
            )

    return report