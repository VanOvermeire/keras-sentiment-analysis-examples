## Keras Machine Learning Examples

This project contains code accompanying my blog post on sentiment analysis with Keras.

[LINK]

### Contents

Three examples are contained in this project.

- a model with dense layers
- a covnet
- a model with GRU layers

The functions folder shows how you might encode files containing your text data and labels for use in
Keras.

To save the tokenizer, pickle was used:

```
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

While for the model itself, I used the build-in save:

`model.save('dutch_reviews_dense.h5')`

### Notes on scraping

- The websites were scraped using a single-threader Python scraper with sleeps between the requests, 
to minimize any pressure on servers
- Robots.txt files were respected
- In deference to the websites, neither scraper nor raw data is included in this project
