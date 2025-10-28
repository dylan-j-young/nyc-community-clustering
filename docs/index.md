---
layout: default
title: NYC Community Clustering
---

# NYC Community Clustering

Welcome! In this project, I explore community structure in New York City using spatial clustering on demographic, housing, and economic data.

## Motivation (150 word goal)

<!-- People often understand the geography of a city in terms of its neighborhoods (see, for example, the New York Times's [An Extremely Detailed Map of New York City Neighborhoods](https://www.nytimes.com/interactive/2023/upshot/extremely-detailed-nyc-neighborhood-map.html)). How well do these named regions map onto how people naturally 

Understanding how a city changes over time requires knowledge of the people who live there, both at a global and a local (neighborhood) level. In particular, studying how neighborhoods grow and change can inform a myriad of topics like gentrification, urban poverty, and politics.

But named neighborhoods are not a complete description of a city's communities! Take Williamsburg as an example: comparing the NYT neighborhood boundary with the results from the 2024 presidential reveals a striking divide:
**(insert figure here: 50% agreement Williamsburg vs. 2024 election map NYT)**

What's going on? The core of Williamsburg is a rapidly gentrifying area filled with **(update) young professionals** who lean progressive and tend to size with Gaza in the Israel-Palestine conflict. Southern Williamsburg, on the other hand, is heavily orthodox Jewish and pro-Israel, leading them to vote Republican due to the party's staunch support for Israel. Both of these communities lie in the area we call "Williamsburg," but grouping them together masks important distinctions in their values and priorities.

My goal is to map out these distinct communities by studying how people of different demographics, housing situations, and income group together into geographic regions using a clustering algorithm. Using these regions, I can evaluate how well the named neighborhoods describe real communities of people, and where this description falls short. 

*If time permits, I will repeat this analysis for past years as well to see how communities change over time. And in the future, I hope to use these regions as a tool to study issues in urban development at the local level.* -->

## Data & Methods (200 word goal)

- **Data Sources:** Describe the datasets you used (e.g., census data, neighborhood boundaries).  
- **Methods:** High-level overview of your clustering approach, features, and any preprocessing. Keep it concise — avoid too many technical details here.

Features:
- (4) Race, ethinicity, and nationality.
    - Four race categories, broken into the fractional population of Hispanic, non-Hispanic white, non-Hispanic black, and other (Asian being the majority, along with Native American/American Indian, Native Hawaiian/Pacific Islander, and multiracial). This only requires three features because these fractions always sum to 1.
    - The fraction of foreign-born residents as a proxy for immigrant communities.
- (2) Age.
    - Median age and the working age population (between 18-64). These can identify strong concentrations of children, adults, and seniors more or less independently.
- (2) Households and socioeconomic status.
    - Median household income. I found that it alone is strongly predictive of several other socioeconomic indicators I proposed, such as median rent and the number of people below the poverty line.
    - The fraction of household owners living with their spouse, providing an axis sensitive to household type.
- (3) Urbanization.
    - Population density, which provides an urban vs. rural spectrum. 
    - The amount of large (20+ unit) apartment buildings and the fraction of people who commute to work by walking. These features identify dense urban cores and hyperlocal communities.

## Results (250 word goal)

<!-- Ooh interactive folium map, fancy -->
<div class="map-container">
  <iframe 
    src="{{ '/maps/map_clusters.html' | relative_url }}"
    loading="lazy"
    allowfullscreen
  ></iframe>
</div>


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

*This page is designed as a concise, ~500–800 word overview of the project. You can expand each section as needed, but the goal is a single-page summary that is easy to navigate.*
