### QUESTIONS

- What factors influence duration of stay
  - # Animal type (dog, cat) --> DALIA
    - columns needed: "Days in Shelter" , "type"
    - multi-Boxplot for each animal type
  - # Breed --> 1 for cat and one 1 for dog --> RAF (DONE)
    - columns needed: "Days in Shelter" - "breed", "type"
    - multi Barplot, one bar for each breed
    - Create function to get single breed, maybe.
  - # Age --> RAF
    - columns needed: "Days in Shelter", "DOB", current date(not a column name!)
    - Scatter plot
    - Cleanup data(unkown DOB), maybe have to put ages in buckets and use barplot instead of scatterplot?

  - # Size (should we spearate species?) separate by age(isPuppy, isKitten)? --> GABE
    - columns needed: "Days in Shelter", "Size" - kitten/puppy or not?
    - Violin or figure it out
    - might have to create new columns - isKitten/isPuppy

  - # Sex (spayed female & neutered male) --> DALIA
    - columns needed: "Days in Shelter", "sex"
    - create column for 'sex_binary'
    

  - # Outcome - "how much does each outcome stay?" --> GABE
    - columns needed: "Outcome Type", "Length of Stay"
    - Barplot
    - seems like clean categories 



------------- Extra Questions --------------------
  - Variable: Intake Condition
  - Length of stay. 
    - Variables: Intake Date, Outcome Date

  - do lighter colors get noticed more? is there is a connection with size?
    - Variable: Color, Size

- Is there a correlation between age and duration?
- Does the time of year affect the duration of stay?
- Do health conditions increase or decrease the length of stay?
