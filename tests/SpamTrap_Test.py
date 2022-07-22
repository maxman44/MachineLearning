examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?", "Give me your password for money!"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)