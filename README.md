# Vectorizing Building Blueprints

Weilian Song, Mahsa Maleki Abyaneh, Mohammad Amin Shabani, Yasutaka Furukawa

_Simon Fraser University_

**ACCV 2022**

![teaser](https://user-images.githubusercontent.com/14151335/205986925-1d6b2d83-e7a0-4b37-a14f-c7f5dc8c2646.png)

**Abstract**: This paper proposes a novel vectorization algorithm for high-definition floorplans with construction-level intricate architectural details, namely a blueprint. A state-of-the-art floorplan vectorization algorithm starts by detecting corners, whose process does not scale to high-definition floorplans with thin interior walls, small door frames, and long exterior walls. Our approach 1) obtains rough semantic segmentation by running off-the-shelf segmentation algorithms; 2) learning to infer missing smaller architectural components; 3) adding the missing components by a refinement generative adversarial network; and 4) simplifying the segmentation boundaries by heuristics. We have created a vectorized blueprint database consisting of 200 production scanned blueprint images. Qualitative and quantitative evaluations demonstrate the effectiveness of the approach, making significant boost in standard vectorization metrics over the current state-of-the-art and baseline methods. We will share our code at https://github.com/weiliansong/blueprint-vectorizer.

[Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Song_Vectorizing_Building_Blueprints_ACCV_2022_paper.pdf)

[Supplementary](https://openaccess.thecvf.com/content/ACCV2022/supplemental/Song_Vectorizing_Building_Blueprints_ACCV_2022_supplemental.zip)

[Code](https://github.com/weiliansong/blueprint-vectorizer)

## Bibtex
```
@InProceedings{Song_2022_ACCV,
    author    = {Song, Weilian and Abyaneh, Mahsa Maleki and A Shabani, Mohammad Amin and Furukawa, Yasutaka},
    title     = {Vectorizing Building Blueprints},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {1044-1059}
}
```

## Acknowledgment
The research is supported by NSERC Discovery Grants, NSERC Discovery Grants Accelerator Supplements, and DND/NSERC Discovery Grants. We also thank GA Technologies for providing us with the building blue-print images.
