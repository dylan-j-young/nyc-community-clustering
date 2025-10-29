---
layout: default
title: NYC Community Clustering
---

# NYC Community Clustering

Welcome! In this project, I explore community structure in New York City using spatial clustering on demographic, housing, and economic data.

## Motivation

<!-- People often understand the geography of a city in terms of its neighborhoods (see, for example, the New York Times's [An Extremely Detailed Map of New York City Neighborhoods](https://www.nytimes.com/interactive/2023/upshot/extremely-detailed-nyc-neighborhood-map.html)). How well do these named regions map onto how people naturally 

Understanding how a city changes over time requires knowledge of the people who live there, both at a global and a local (neighborhood) level. In particular, studying how neighborhoods grow and change can inform a myriad of topics like gentrification, urban poverty, and politics.

But named neighborhoods are not a complete description of a city's communities! Take Williamsburg as an example: comparing the NYT neighborhood boundary with the results from the 2024 presidential reveals a striking divide:
**(insert figure here: 50% agreement Williamsburg vs. 2024 election map NYT)**

What's going on? The core of Williamsburg is a rapidly gentrifying area filled with **(update) young professionals** who lean progressive and tend to size with Gaza in the Israel-Palestine conflict. Southern Williamsburg, on the other hand, is heavily orthodox Jewish and pro-Israel, leading them to vote Republican due to the party's staunch support for Israel. Both of these communities lie in the area we call "Williamsburg," but grouping them together masks important distinctions in their values and priorities.

My goal is to map out these distinct communities by studying how people of different demographics, housing situations, and income group together into geographic regions using a clustering algorithm. Using these regions, I can evaluate how well the named neighborhoods describe real communities of people, and where this description falls short. 

*If time permits, I will repeat this analysis for past years as well to see how communities change over time. And in the future, I hope to use these regions as a tool to study issues in urban development at the local level.* -->

<!-- Remember to mention that you're performing SPATIAL clustering on NYC CENSUS TRACTS. -->

## Gathering data

<!-- I pulled demographic data primarily from the 2020 US Census Demographic Profile, supplemented with economic and urbanization data from the 2023 5-year American Community Survey.  -->
I pulled data from the 2020 US Census Demographic Profile and the 2023 5-year American Community Survey. Using a combination of PCA, linear and nonlinear regression analysis, and spatial autocorrelation checks, I chose 11 largely independent features exhibiting interesting spatial structure. They cover a wide gamut of information:
- (4) Race and nationality.
    - Fractional populations of Hispanic, non-Hispanic white, non-Hispanic black, and other (Asian being the majority, along with Native American, Pacific Islander, and multiracial populations). This is only three independent features because the fractions sum to 1.
    - The fraction of foreign-born residents. <!-- as a proxy for immigrant communities. -->
- (2) Age.
    - Median age and fractional working age population (between 18-64). <!-- These can identify strong concentrations of children, adults, and seniors more or less independently. -->
- (2) Household type and socioeconomic status.
    - Median household income. <!-- I found that it alone is strongly predictive of several other socioeconomic indicators I proposed, such as median rent and the number of people below the poverty line. -->
    - The fraction of household owners living with their spouse. <!-- , providing an axis sensitive to household type. -->
- (3) Urbanization.
    - Population density. <!-- , which provides an urban vs. rural spectrum. -->
    - The amount of large (20+ unit) apartment buildings and the fraction of people who commute to work by walking. <!-- These features identify dense urban cores and hyperlocal communities. -->


## Using a spatially aware clustering algorithm

When clustering geographies into local regions, it is vital to use an algorithm that takes geography into account. Common methods like K-Means only consider feature space and produce clusters that group spatially distant regions together:

<!-- K-Means figure -->
<figure class="responsive-figure">
    <img 
        src="{{ '/figures/kmeans_clusters.png' | relative_url }}"
        alt="K-Means clustering on NYC Census tracts, with K=10 clusters."
    />
    <figcaption>
        Each color represents one cluster. I'd rather not group parts of Staten Island and Queens together, even if they're demographically similar.
    </figcaption>
</figure>

<!-- It is a two-step method that first connects Census tracts together to form a tree, taking both feature similarity and spatial connectivity into account, and then prunes the tree into the desired number of clusters.  -->
Instead, I used an algorithm belonging to a family of methods called REDCAP<sup><a href="#ref-redcap">1</a></sup>, which explicitly require clusters to respect contiguity. This algorithm performed the best on a combination of metrics, chosen to evaluate feature similarity and geographic compactness simultaneously:
- The Polsby-Popper index (tests for compactness)
- The Davies-Bouldin index (tests for feature similarity)
- The "path" silhouette<sup><a href="#ref-pathsilhouette">2</a></sup> (combination of both). 
    - I actually modified the metric to fix some perceived issues (documented in the project notebooks for anyone who's curious).

Check out the results in the map below!

<!-- Ooh interactive folium map, fancy -->
<figure class="responsive-figure map-figure">
  <div class="map-container">
    <iframe 
        src="{{ '/maps/map_clusters.html' | relative_url }}"
        title="NYC demographic clusters map"
        loading="lazy"
        allowfullscreen
    ></iframe>
  </div>
  <figcaption>Interactive map of NYC cluster assignments.</figcaption>
</figure>

## What can we learn from demographic clustering?

Summarize key findings:
- Number of clusters identified
- Notable patterns or trends
- Any interesting visualizations (consider embedding images with `![alt text](assets/figure.png)`)

## Conclusions (100 word goal)

*Highlight the takeaways from your analysis:*
- *What did the clustering reveal about NYC communities?* 
- *Any implications or next steps?*

Potential ways to use this data:

- Using these clusters as a geographic filter for city-wide data can reveal community-level needs, issues, and inequities.
    - Food deserts, transit deserts, demand for businesses, demand for city services at a community level
(why is community level so important? Need to clarify my thoughts on this.)

---

## References

<ol class="references">
    <li id="ref-redcap">
        Guo, D. (2008). 
        <em>Regionalization with dynamically constrained agglomerative clustering and partitioning (REDCAP).</em>
        International Journal of Geographical Information Science.
    </li>
    <li id="ref-pathsilhouette">
        Wolf, L. J., Knaap, E., and Rey, S. (2019). 
        <em>Geosilhouettes: Geographical measures of cluster fit.</em>
        Environment and Planning B: Urban Analytics and City Science.
    </li>
</ol>
