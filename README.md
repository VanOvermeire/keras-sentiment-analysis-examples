## Keras Machine Learning Examples

This project contains code accompanying my blog post on sentiment analysis with Keras.

### Contents

Three examples are contained in this project.

- a dense model
- a covnet
- a GRU model

The functions folder shows how you might encode files containing your text data and labels for use in
Keras.

To save the tokenizer, pickle was used:

```
with open('example_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

While for the model itself, I used the build-in save method:

`model.save('example_saved_model.h5')`

#### Notes on scraping

- The websites were scraped using a single-threader Python scraper with sleeps between the requests, 
to minimize any pressure on servers.
- Robots.txt files were respected.
- In deference to the websites, neither scraper nor raw data is included in this project.

#### TODO

- Add saved models and pickles