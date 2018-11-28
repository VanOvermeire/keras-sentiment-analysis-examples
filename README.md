## Keras Machine Learning Examples

This project contains code accompanying my blog post on sentiment analysis with Keras. 

The code is based both on [official keras examples][1] and the book [Deep Learning With Python][2]

[1]: https://github.com/keras-team/keras/tree/master/examples
[2]: https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438

### Contents

This project contains:

- a dense model
- a covnet
- a GRU model

For sentiment analysis. It also contains code for creating a text-generation model.

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
