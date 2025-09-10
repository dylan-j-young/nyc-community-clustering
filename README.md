# Project Title: Mapping NYC Communities

Clustering NYC into communities based on census demographics, housing data, and income.

---

## 📊 Project Overview

### Motivation

NOTE: this needs heavy editing. Start by looking at my convo with ChatGPT for inspiration on making this more approachable and recruiter-friendly.

Understanding how a city changes over time requires knowledge of the people who live there, both at a global and a local (neighborhood) level. In particular, studying how neighborhoods grow and change can inform a myriad of topics like gentrification, urban poverty, and politics.

In 2023, the New York Times published [An Extremely Detailed Map of New York City Neighborhoods](https://www.nytimes.com/interactive/2023/upshot/extremely-detailed-nyc-neighborhood-map.html), along with a fascinating article discussing the fluid nature of these regions. Their study was based on a NYT survey asking residents to draw boundaries of their local neighborhoods, along with their names (such as Bed-Stuy, Williamsburg, and the more controversial "East Williamsburg"). 

But named neighborhoods are not a complete description of a city's communities! Take Williamsburg as an example: comparing the NYT neighborhood boundary with the results from the 2024 presidential reveals a striking divide:
**(insert figure here: 50% agreement Williamsburg vs. 2024 election map NYT)**

What's going on? The core of Williamsburg is a rapidly gentrifying area filled with **(update) young professionals** who lean progressive and tend to size with Gaza in the Israel-Palestine conflict. Southern Williamsburg, on the other hand, is heavily orthodox Jewish and pro-Israel, leading them to vote Republican due to the party's staunch support for Israel. Both of these communities lie in the area we call "Williamsburg," but grouping them together masks important distinctions in their values and priorities.

My goal is to map out these distinct communities by studying how people of different demographics, housing situations, and income group together into geographic regions using a clustering algorithm. Using these regions, I can evaluate how well the named neighborhoods describe real communities of people, and where this description falls short. 

*If time permits, I will repeat this analysis for past years as well to see how communities change over time. And in the future, I hope to use these regions as a tool to study issues in urban development at the local level.*

- **Approach:** *A quick summary of your methodology (e.g., "Exploratory data analysis + regression modeling").*  

- **Result:** *Tease your key findings (e.g., "Achieved R² of 0.82 on test set").* 

---

## 📂 Repository Structure

My repository structure is inspired by `cookiecutter-data-science` [project link](https://cookiecutter-data-science.drivendata.org/).


...

DISCLAIMER: This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau.

To pull data from the Census API, it's recommended to request an API key (limited queries are available without one, but frequent queries require a key). You can make this request at [https://www.census.gov/data/developers.html](https://www.census.gov/data/developers.html).

Once you get your key, create a file called `.env` in the project root with the following line:
```
CENSUS_API_KEY=your_key_here
```