# Kappa-IoU

This Python package implements the Kappa-IoU measure of Inter-annotator reliability from Rogers (2026):

    Rogers, Trevor F. (2026). Kappa-IoU: Inter-Rater Reliability for Spatial Annotation
       
Large-scale annotation projects for which automated annotation solutions are insufficient require a metric that measures the quality of annotation between annotators. 
[Intersection-over-Union](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=Rezatofighi%2C%20H.%2C%20Tsoi%2C%20N.%2C%20Gwak%2C%20J.%2C%20Sadeghian%2C%20A.%2C%20Reid%2C%20I.%2C%20%26%20Savarese%2C%20S.%20(2019).%20Generalized%20intersection%20over%20union%3A%20A%20metric%20and%20a%20loss%20for%20object%20detection.%20Proceedings%20of%20the%20IEEE%2FCVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20(CVPR).), or IoU, is traditionally used for spatial data but requires a ground truth. 
[Cohen's Kappa](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=Cohen%2C%20J.%20(1960).%20A%20coefficient%20of%20agreement%20for%20nominal%20scales.%C2%A0Educational%20and%20psychological%20measurement%2C%C2%A020(1)%2C%2037-46.) is a psychometric measure that measures the amount of agreement between raters, but is traditionally used for categorical data. This repository contains the code recreating the results of section 4 of the [Kappa-IoU](https://drive.google.com/file/d/1QW1dbFWX1-rxz7XcvPkBdlR0BFKoBIwC/view?usp=sharing) paper, which indexes IoU with Cohen's Kappa, handles differences in object cardinality between raters by treating it as a linear assignment problem using [Kuhn-Munkres](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=Munkres%2C%20J.%20(1957).%20Algorithms%20for%20the%20assignment%20and%20transportation%20problems.%20Journal%20of%20the%20Society%20for%20Industrial%20and%20Applied%20Mathematics%2C%205(1)%2C%2032-38.) assignment. An adjusted version of Kappa-IoU and uses a version of the [BLEU](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=Papineni%2C%20K.%2C%20Roukos%2C%20S.%2C%20Ward%2C%20T.%2C%20%26%20Zhu%2C%20W.%20J.%20(2002).%20BLEU%3A%20a%20method%20for%20automatic%20evaluation%20of%20machine%20translation.%20In%20Proceedings%20of%20the%2040th%20Annual%20Meeting%20of%20the%20Association%20for%20Computational%20Linguistics%20(pp.%20311-318).) measure for machine translation which scores N-objects for multi-object annotation.

To install from a inside the downloaded folder from Github:

    python -m pip install -e .

To use, run the analysis engine straight from your terminal shell across your dataset directories using the console entrypoint:

    kappaiou-run --baseline /data/author_baseline.csv --dir /data/annotator_tranches/ --tau 0.70

The author baseline and annotator tranche CSV files are expected to have the filenames in the first column, label in the second column, and then xywh-format annotations in the following four columns.

MIT License
