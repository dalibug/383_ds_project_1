# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="aab1aa45"
# #  Visualizing Adoption and Return Trends in Sonoma Animal Data
# 03/15/2024

# %% [markdown] id="8d17ef9f"
# #### Rafael L.S Reis, Dalia Cabrera Hurtado, Gabriel Myers

# %% [markdown] id="dd9d8a1b"
# ## Introduction
# The Sonoma Animal Shelter dataset, provided by the County of Sonoma Department of Health Services, comprises about 30,000 records detailing various attributes of animals admitted to the shelter. Each record includes species, breed, age, sex, and color, along with intake and outcome types to track the animals journey in the shelter. With over 4,000 animals entering the shelter annually, the dataset offers insights into adoption trends, shelter capacity, and animal welfare efforts. It gets updated regularly with the most recent being March 18, 2025.
#
# This analysis seeks to answer two key questions: first, how does the number of days an animal spends in the shelter differ between those that are adopted and those that are returned to their owners; and second, is there an association between an animal's primary coat color—extracted from compound color entries—and its outcome or duration of shelter stay.
#
# note: This is tentative and we will likely hone down our scope of questionings(likely to colors)
#
# data downloaded from:
# https://raw.githubusercontent.com/grbruns/cst383/master/sonoma-shelter-15-october-2024.csv

# %% id="fdf3ed81"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown] id="f03dd8de"
# ## Initial Data Exploration

# %% id="8694f864"
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/sonoma-shelter-17-march-2025.csv')

# %% [markdown] id="d1283d9a"
# ## Data preprocessing

# %% id="85iYy59zSfzm" outputId="b0cffc8d-33bc-4a95-a8b7-02511d982bfc" colab={"base_uri": "https://localhost:8080/"}
df.info()

# %% id="lNr84NVtSjtG" outputId="f2363223-5fcd-4c64-8939-4eb98d3cbf3f" colab={"base_uri": "https://localhost:8080/", "height": 714}
df.sample(10)


# %% [markdown] id="K1CCaeqrSsuf"
# Among dogs returned to their owners, which coat colors are most frequently associated with being lost or having escaped? To try explore this, we can create a simplified color category column to analyze color hues effectively.

# %% [markdown] id="5clMunrXW8SN"
# Below are the functions to help create a new column to get the color shade of the animals

# %% id="olpOT58oT0WN"
# Function to get primary color from a color string
def get_primary_color(color):
    if pd.isnull(color):
        return 'unknown'
    return color.split('/')[0].strip().lower()

# Helper function to check if primary color is in a given list, maybe not necessary but cleaner?
def primary_color_in_list(primary_color, shades_list):
    return any(shade in primary_color for shade in shades_list)

# Function to categorize color into Light, Medium, Dark, or Other shades
def categorize_shade(color):
    if pd.isna(color):
        return 'Unknown'

    primary_color = get_primary_color(color)
    # mappings, might play around with these more...
    dark_shades = ['black', 'brown', 'brindle', 'blue', 'gray', 'chocolate', 'seal']
    medium_shades = ['tan', 'red', 'gold', 'fawn', 'sable', 'yellow', 'orange']
    light_shades = ['white', 'cream', 'buff']

    if primary_color_in_list(primary_color, dark_shades):
        return 'Dark'
    elif primary_color_in_list(primary_color, medium_shades):
        return 'Medium'
    elif primary_color_in_list(primary_color, light_shades):
        return 'Light'
    else:
        return 'Other'



# %% [markdown] id="SOMAFVapXsnL"
# Applying filter to create column

# %% colab={"base_uri": "https://localhost:8080/"} id="86v1ug83Xu0k" outputId="edd3ad89-8330-4369-8e69-8438d4174ff1"
dogs_returned = df[
    (df['Type'] == 'DOG') &
    (df['Outcome Type'].str.upper() == 'RETURN TO OWNER')
].copy()

# Create 'Primary Color' column
dogs_returned['Primary Color'] = dogs_returned['Color'].apply(get_primary_color)

# Create 'Primary Shade' column correctly
dogs_returned['Primary Shade'] = dogs_returned['Primary Color'].apply(categorize_shade)

# just sampling to check if things look about right
print(dogs_returned[['Name', 'Color', 'Primary Color', 'Primary Shade']].head(10))


# %% [markdown] id="5a3a9d08"
# ## Data exploration and visualization

# %% [markdown] id="QOJShgqBU5m-"
# Exploring data from primary color column

# %% colab={"base_uri": "https://localhost:8080/", "height": 513} id="QR9EFrXnU-13" outputId="cad25fca-e367-485d-90dc-ff2a33182fb5"
# Plot the shade distribution
shade_counts = dogs_returned['Primary Shade'].value_counts()
shade_counts.plot(kind='bar')

plt.title('Distribution of Coat Shades for Dogs Returned to Owners')
plt.xlabel('Shade Category')
plt.ylabel('Number of Dogs')
plt.show()

# %% [markdown] id="oswY20dtVAhI"
# Data above looks good(at least we can see the relative primary colors), but doesn't give us the full picture, maybe there's just more black dogs. Let's explore some more. Out of all dogs of a given shade, what's the proportion successfully returned to their owner?

# %% [markdown]
# ## Size

# %% [markdown]
#  - #Size (should we spearate species?) separate by age(isPuppy, isKitten)? GABE
#     - columns needed: "Days in Shelter", "Size" - kitten/puppy or not?
#     - Violin or figure it out
#     - might have to create new columns - isKitten/isPuppy

# %%
# create a new column to tell if an animal is a puppy or kitten
df["is_puppy_kitten"] = (df["Size"] == "KITTEN") | (df["Size"] == "PUPPY")
# create new column for log and use log1p to handle zeros
df["Days in Shelter_log"] = np.log1p(df["Days in Shelter"]) 
sns.violinplot(x="is_puppy_kitten", y="Days in Shelter_log", data=df)
plt.title("Days in Shelter for Puppies and Kittens (Log-Transformed)")
plt.xlabel("Is Puppy or Kitten")
plt.ylabel("Days in shelter (log scale)")
plt.show();

# %% [markdown]
# These two distributions look pretty simialar execept there is a spike close to 0 for non puppies and kittens. I think this is from peoples animals getting picked up off the street and returned to the owner.

# %% [markdown]
# ## Outcome

# %% [markdown]
# - #Outcome - "how much does each outcome stay?" --> GABE
#     - columns needed: "Outcome Type", "Length of Stay"
#     - Barplot
#     - seems like clean categories 

# %%
df_group = df.groupby("Outcome Type")["Days in Shelter"].mean()
sns.barplot(x="Outcome Type", y="Days in Shelter", df=df_group)
plt.xlabel("Outcome Type")
plt.ylabel("Average Days in Shelter")
plt.title("Average Days in Shelter by Outcome Type")
plt.xticks(rotation=45)
plt.show()

# %%
df.groupby("Outcome Type")["Days in Shelter"].mean().sort_values(ascending=False).plot.bar()
plt.xlabel("Outcome Type")
plt.ylabel("Average Days in Shelter")
plt.title("Average Days in Shelter by Outcome Type")
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# As we can see adopted animals stay the longest with an average of 40 days. This suggests that adoption is a proccess and takes time. I would also assume that the shelter staff are trying to find the best match possible for each dog so some people might get turned away.
# Another interesting thing to note is how quickly animals get returned to their owner on average its about 4 days.

# %% [markdown] id="05ea9473"
# ## Conclusions

# %% [markdown] id="gczUjeHafqXM"
# For now we still need to explore more and improve the notebook. As it stands it's pretty messy but we just wanted to explore as much as we could first and see if we found anything of interest or significance rather than caring too much about form. As we hone down on our areas of interest we will make the data look better and have better descriptions and organization. Lastly exploring the effects of color might be more interesting(given our exploration) so we might pivot to focus more on that.
