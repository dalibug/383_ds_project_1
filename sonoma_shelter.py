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
# #### Rafael L.S Reis, Dalia Cabrera Hurtado, Gabe Myers

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
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/sonoma-shelter-15-october-2024.csv')

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

# %% colab={"base_uri": "https://localhost:8080/", "height": 207} id="tdgI649qVLXs" outputId="1128c834-33ca-49c7-afa7-8a4979588d51"
# Calculate the total count and returned count per shade
shade_summary = dogs_df.groupby('Primary Shade')['Returned'].agg(['sum', 'count'])
shade_summary['Returned Proportion'] = shade_summary['sum'] / shade_summary['count']
shade_summary = shade_summary.sort_values('Returned Proportion')

#print(shade_summary)

#proportion of returned dogs by shade
plt.figure(figsize=(8,6))
plt.bar(shade_summary.index, shade_summary['Returned Proportion'])
plt.title('Proportion of Dogs Returned to Owner by Coat Shade')
plt.xlabel('Coat Shade')
plt.ylabel('Proportion Returned')
plt.show()


# %% [markdown] id="HJL2Usawa8S8"
# There seems to be a difference, but slight, is it enough to draw any conlusions? Do i need to clean the data better so we dont have 'Other'?

# %% [markdown] id="_XDOGUt0dOfh"
# Other exploration ideas: Time in shelter - Dogs who got returned vs not returned:

# %% colab={"base_uri": "https://localhost:8080/", "height": 207} id="8x5Yu-r3a-qx" outputId="ec53b2f8-001e-49d6-e23f-ab56b6963bdb"
dogs_df['Returned'] = (dogs_df['Outcome Type'].str.upper() == 'RETURN TO OWNER').astype(int)
#print(dogs_df[['Primary Shade', 'Returned']].head())

sns.violinplot(
    data=dogs_df,
    x='Returned',
    y='Days in Shelter',
    hue='Primary Shade',
    # split=True,
    inner='quartile',
    cut=0  # so the violin doesn't extend beyond actual data
)
plt.yscale('log')  # Compress large values
plt.title('Distribution of Days in Shelter (Log Scale)')
plt.xlabel('Returned Status (0 = Not Returned, 1 = Returned)')
plt.ylabel('Days in Shelter (log scale)')
plt.legend(title='Coat Shade', loc='upper right')
plt.tight_layout()
plt.show()

# %% [markdown] id="78cGnub7fx7n"
# We're uncertain how relevant this is, needs further investigation and better plotting.

# %% [markdown] id="Dzp5QNGsgCe1"
# ### Adoptet vs Found owner
# Trying to improve or get a different angle from the above violin plot

# %% id="xar0vIk3gKx5"
# Filter dataset to include only dogs that were either adopted or returned to owner
dogs_subset = df[
    (df['Type'] == 'DOG') &
    (df['Outcome Type'].str.upper().isin(['ADOPTION', 'RETURN TO OWNER']))
].copy()

# 2. Standardize the Outcome Type to make data cleaner, was haivng issues
dogs_subset['Outcome Type'] = dogs_subset['Outcome Type'].str.upper()

#boxplot for Days in Shelter by Outcome Type
sns.boxplot(data=dogs_subset, x='Outcome Type', y='Days in Shelter')
plt.title('Comparing Days in Shelter: Adopted vs. Returned to Owner')
plt.xlabel('Outcome Type')
plt.ylabel('Days in Shelter')
plt.tight_layout()
plt.show()

# %% [markdown] id="rBXU-9yurjh5"
# We need to get a better fit for this graph, and while stil not ideal, the violin graph seems better than this one to represent days in shelter x Returned vs not returned

# %% [markdown] id="on3zYlwikfXv"
# ###Exploring colors - how long they take to get adopted

# %% id="MUMiFzy_gvzC"
common_colors = dogs_df['Primary Color'].value_counts()
common_colors = common_colors[common_colors >= 30].index.tolist()
common_color_df = dogs_df[dogs_df['Primary Color'].isin(common_colors)]
filtered_df = common_color_df[common_color_df['Days in Shelter'] <= 60]

plt.figure(figsize=(12, 8))
ax = sns.boxplot(
    data=filtered_df,
    x='Primary Color',
    y='Days in Shelter',
    hue='Primary Color',
    palette='Set1'
)

plt.title('Distribution of Days in Shelter (< 100 Days) by Primary Color\n(Common Colors Only)')
plt.xlabel('Primary Color')
plt.ylabel('Days in Shelter')
plt.xticks(rotation=45)
plt.show()

# %% [markdown] id="3ZKxDI-zkoEM"
# It seems certain colors (silver, gold and yellow) are adopted faster than avarage whilst other take longer(tan, blue, buff). We might need to come p with a better way to represented colors.

# %% colab={"base_uri": "https://localhost:8080/", "height": 704} id="SrMSG7GwtfrW" outputId="bd8b2852-f501-4547-c370-a230a1acb19f"
df["Intake Date"] = pd.to_datetime(df["Intake Date"])
df["Outcome Date"] = pd.to_datetime(df["Outcome Date"])
df["length_of_stay"] = (df["Outcome Date"] - df["Intake Date"]).dt.days

df["Outcome Type"] = df["Outcome Type"].str.strip().str.lower()
df_return = df[df["Outcome Type"] == "return to owner"]

df_return["Type"] = df_return["Type"].str.strip().str.lower()
df_return = df_return[df_return["Type"].isin(["cat", "dog"])]

animal_types = df_return["Type"].unique()

fig, axs = plt.subplots(1, len(animal_types), figsize=(12, 6), sharex=True, sharey=True)
if len(animal_types) == 1:
    axs = [axs]

for i, animal in enumerate(animal_types):
    data = df_return[df_return["Type"] == animal]["length_of_stay"]
    data.plot.density(ax=axs[i], label=animal)
    axs[i].set_title(f"Density for {animal.capitalize()}", fontweight='bold')
    axs[i].set_xlabel("Length of Stay (days)")
    axs[i].set_ylabel("Density")
    axs[i].set_xlim(-10, 50)
    axs[i].tick_params(axis='y', labelleft=True, labelright=False)

plt.suptitle("Density of Length of Stay for Animals Returned to Owner by Type", fontweight='bold', fontsize=20)
plt.show()

# %% [markdown] id="wHdO3JRYuVsO"
# The data suggest that the duration of an animal's stay is strongly associated with whether it is returned to its owner, implying that length of stay may be a useful predictor of this outcome.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 578} id="3GYXQk4NvIsY" outputId="ede735d1-a555-4d18-bafa-8aaa80d99cd6"
df["Intake Date"] = pd.to_datetime(df["Intake Date"])
df["Outcome Date"] = pd.to_datetime(df["Outcome Date"])

df["length_of_stay"] = (df["Outcome Date"] - df["Intake Date"]).dt.days

df["Outcome Type"] = df["Outcome Type"].str.strip().str.lower()

df_return = df[df["Outcome Type"] == "return to owner"]

df_return["Type"] = df_return["Type"].str.strip().str.lower()
df_return = df_return[df_return["Type"].isin(["cat", "dog"])]
df_return = df_return[(df_return['Days in Shelter'] <= 60)]

plt.figure(figsize=(8, 6))
df_return.boxplot(column="length_of_stay", by="Type", grid=False)
plt.title("Length of Stay for Animals Returned to Owner by Animal Type", fontweight='bold', fontsize=10)
plt.suptitle("")
plt.xlabel("Animal Type")
plt.ylabel("Length of Stay (days)")
plt.show()

# %% [markdown] id="qaqR4FKVvKUa"
# It seems that cat owners will give up on retrieving their animal faster than dog owners...

# %% [markdown] id="05ea9473"
# ## Conclusions

# %% [markdown] id="gczUjeHafqXM"
# For now we still need to explore more and improve the notebook. As it stands it's pretty messy but we just wanted to explore as much as we could first and see if we found anything of interest or significance rather than caring too much about form. As we hone down on our areas of interest we will make the data look better and have better descriptions and organization. Lastly exploring the effects of color might be more interesting(given our exploration) so we might pivot to focus more on that.
