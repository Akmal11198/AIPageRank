import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # if page has no links
    if len(corpus[page]) == 0:
        result = dict()
        prob = float(1/len(corpus))
        for p in corpus:
            result[p] = prob
        return result

    # page has links
    result = dict()
    # first add 1-damping_factor/N to all pages
    for p in corpus:
        prob = float((1-damping_factor)/len(corpus))
        result[p] = prob

    # add damping factor/N of links to linked pages
    prob = float(damping_factor/len(corpus[page]))
    for p in corpus[page]:
        result[p] += prob
    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    result = dict()
    t_model = dict()
    # first transition model is set to 1 for all pages to get a random page
    # result is set to zero for all pages
    for p in corpus:
        result[p] = 0
        t_model[p] = 1

    # sample
    for _ in range(n):
        current_page = random.choices(
            list(corpus.keys()), weights=t_model.values(), k=1)[0]
        result[current_page] += 1
        t_model = transition_model(corpus, current_page, damping_factor)

    for p in result:
        result[p] /= n
    return result


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    result = dict()
    # initially all pages are assigned 1/N
    for p in corpus:
        result[p] = float(1/N)

    while True:
        go = False
        for page in result:
            sigma = 0
            # looking for pages that link to our current page
            for linker in corpus:
                if len(corpus[linker]) == 0:
                    sigma += result[linker]/len(corpus)
                elif page in corpus[linker]:
                    sigma += result[linker]/len(corpus[linker])

            new_result = (1-damping_factor)/N+damping_factor*sigma
            # comparing to see if change is significant or stop iteration
            if math.fabs(new_result-result[page]) > 0.001:
                go = True
            result[page] = new_result
        if not go:
            break
    return result


if __name__ == "__main__":
    main()
