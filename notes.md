## Structure of the data set
- The folder Gallery contains n=3531 samples (dim=1024) which represent unique identities (=template).
- The folder Probes contains n=19557 different samples (dim=1024) of the identities in the gallery.
- The file identities.txt shows a unique mapping of all samples (gallery and probe) to a gallery sample (by index). So each probe sample belongs to one gallery sample and there are no probe samples that are not corresponding to a gallery sample (= closed-set).
- The file distances.txt shows the distance (col4) from every probe sample (col2) to the corresponding gallery sample (col1) and gallery index (col3).

## Similarity measures
- Cosine similarity (-1 to 1): -1 -> opposite directions, 1 -> same direction
- Cosine distance = 1 - cosine similarity
- Cosine distance (0 to 2): 0 -> same direction, 2 -> opposite directions
- L2 distance (Euklidian)
- Jaccard similarity

## Performance metrics
- Rank 1 identification rate (-> accuracy/hit rate): `#correctly_classified_samples / #classified_samples`
- Rank 5 identification rate: `#correctly_classified_within_top_5_samples / #classified_samples`
- "The hit rate is the portion of the searches where the correct identity is found within the considered percentage of the references out of the complete database (penetration rate)" Damer et al. 2017 -> What is the penetration rate if the correct identity is not found?

## Related Work
- Damer et al. (https://www.researchgate.net/publication/318729958_Indexing_of_Single_and_Multi-instance_Iris_Data_Based_on_LSH-Forest_and_Rotation_Invariant_Representation)

## References
- LSH (http://mlwiki.org/index.php/Locality_Sensitive_Hashing)
- LSH (https://randorithms.com/2019/09/19/Visual-LSH.html)
- LSH (https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23)

## Approaches/Ideas
- Use PCA for dimensionality reduction (normalize the data prior to that) and compare the projected features with an epsilon tolerance -> first that is similar enough is the prediction
- Use LSH-Forest as proposed by Damer et al.
- Use LSH with different Hashfunction families

## Locality Sensitive Hashing (LSH)
- Hamming LSH (Bit Sampling)
- Signed-/ Random Projections LSH
- Hyperplane LSH
- Euclidean (E2) and Manhattan LSH
- Clustering LSH
- K-Means LSH
- Bayesian LSH
- Random Binary Projection
- SimHash

## Enhancements for LSH
- Additional dimensionality reduction with PCA?
- Reducing number of Hash Tables in LSH (Entropy LSH, Multi-Probe LSH, Distributed Layered LSH)

## Other
- Python library for Random Projections LSH (https://pypi.org/project/lshashing/)
- Shazam uses LSH concept
