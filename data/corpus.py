"""
Rich default document corpus covering multiple distinct domains:
Computer Science & AI, Information Retrieval, Space & Astronomy, Climate & Earth, Medicine & Health, History.
"""
from typing import List, Dict

SAMPLE_DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "DOC01",
        "title": "Introduction to Information Retrieval and Vector Space Models",
        "category": "Information Retrieval",
        "content": (
            "Information retrieval (IR) is the science of searching for documents and metadata in collections. "
            "The vector space model represents text documents and user queries as multidimensional vectors in a shared "
            "feature space. Term Frequency (TF) measures how often a word occurs in a document, while Inverse Document "
            "Frequency (IDF) discounts terms that occur across many documents. Cosine similarity calculates the cosine "
            "of the angle between query and document vectors to quantify document relevance."
        )
    },
    {
        "id": "DOC02",
        "title": "Relevance Feedback and the Rocchio Algorithm in Search Engines",
        "category": "Information Retrieval",
        "content": (
            "Relevance feedback improves search accuracy by allowing users to tag retrieved items as relevant or non-relevant. "
            "The Rocchio algorithm computes a modified query vector by adding the normalized vector centroid of relevant documents "
            "and subtracting the centroid of irrelevant documents. This pushes the query representation closer to desired documents "
            "and pulls it away from false positives, dramatically improving precision in subsequent ranking rounds."
        )
    },
    {
        "id": "DOC03",
        "title": "Boolean Retrieval and Inverted Index Architectures",
        "category": "Information Retrieval",
        "content": (
            "Boolean retrieval systems evaluate user queries using logical operators such as AND, OR, and NOT. "
            "An inverted index maps each vocabulary token to a postings list containing document identifiers. "
            "Queries combining Boolean conditions like 'machine AND learning OR intelligence' are evaluated by intersecting "
            "or unioning the postings lists. Modern hybrid search engines combine strict Boolean filtering with TF-IDF ranking."
        )
    },
    {
        "id": "DOC04",
        "title": "Deep Learning and Transformer Architectures in Natural Language Processing",
        "category": "Artificial Intelligence",
        "content": (
            "Deep neural networks and Transformer architectures revolutionized natural language processing (NLP). "
            "Self-attention mechanisms allow neural models to weigh relationships between distant words across a sentence. "
            "Large language models pre-trained on massive text corpora perform few-shot semantic retrieval, translation, "
            "summarization, and reasoning tasks with unprecedented fluency."
        )
    },
    {
        "id": "DOC05",
        "title": "Supervised Machine Learning and Classification Algorithms",
        "category": "Artificial Intelligence",
        "content": (
            "Supervised machine learning algorithms learn predictive mapping functions from labeled training datasets. "
            "Popular classification algorithms include Support Vector Machines (SVM), Random Forests, Gradient Boosted Decision "
            "Trees, and Logistic Regression. Model evaluation relies on precision, recall, F1-score, and cross-validation matrices."
        )
    },
    {
        "id": "DOC06",
        "title": "Reinforcement Learning and Autonomous Intelligent Agents",
        "category": "Artificial Intelligence",
        "content": (
            "Reinforcement learning trains artificial agents to maximize cumulative scalar rewards through trial-and-error "
            "interactions with complex environments. Techniques like Q-learning, Policy Gradients, and Actor-Critic networks "
            "power autonomous robotic controllers, game playing agents like AlphaGo, and automated trading algorithms."
        )
    },
    {
        "id": "DOC07",
        "title": "James Webb Space Telescope Discoveries in Early Cosmology",
        "category": "Astronomy & Space",
        "content": (
            "The James Webb Space Telescope (JWST) uses high-resolution infrared sensors to peer into the deep cosmos. "
            "JWST captured spectral observations of early galaxies that formed merely hundreds of millions of years after the Big Bang. "
            "Astronomers analyze the redshift of distant stars and investigate atmospheric chemical signatures of exoplanets."
        )
    },
    {
        "id": "DOC08",
        "title": "Mars Exploration: Rovers, Water Evidence, and Planetary Geology",
        "category": "Astronomy & Space",
        "content": (
            "NASA robotic rovers including Curiosity and Perseverance explore Martian craters to study ancient habitability. "
            "Sedimentary rock layers, delta fan formations, and mineral salts provide compelling evidence of past liquid water on Mars. "
            "Soil sample collection and radar instruments examine subsurface ice deposits for future crewed exploration missions."
        )
    },
    {
        "id": "DOC09",
        "title": "Global Climate Change, Renewable Energy, and Carbon Neutrality",
        "category": "Earth & Environment",
        "content": (
            "Anthropogenic carbon dioxide emissions accelerate global temperature rise and ocean acidification. "
            "Transitioning from fossil fuels to clean renewable energy sources like solar photovoltaic panels, offshore wind farms, "
            "and geothermal power is essential to achieve carbon neutrality and mitigate catastrophic extreme weather disasters."
        )
    },
    {
        "id": "DOC10",
        "title": "Oceanic Ecosystems, Coral Bleaching, and Marine Biodiversity",
        "category": "Earth & Environment",
        "content": (
            "Marine ecosystems support vast global biodiversity and regulate climate cycles. Rising sea surface temperatures cause "
            "widespread coral bleaching across barrier reefs. Protecting marine habitats from commercial overfishing, plastic pollution, "
            "and toxic runoff preserves vital food webs and coastal defenses."
        )
    },
    {
        "id": "DOC11",
        "title": "mRNA Vaccines, Immunology, and Modern Biomedical Therapeutics",
        "category": "Medicine & Healthcare",
        "content": (
            "Messenger RNA (mRNA) vaccine platforms deliver synthetic genetic instructions inside lipid nanoparticles to teach "
            "human cells how to produce harmless viral spike antigens. This triggers robust cellular and antibody immune responses, "
            "catalyzing new therapeutic frontiers in infectious disease prevention and personalized cancer immunotherapy."
        )
    },
    {
        "id": "DOC12",
        "title": "Genomics, CRISPR Cas9 Gene Editing, and Precision Medicine",
        "category": "Medicine & Healthcare",
        "content": (
            "CRISPR-Cas9 gene editing allows molecular biologists to make precise molecular alterations in DNA sequences. "
            "Combined with high-throughput genome sequencing, targeted gene therapy addresses hereditary genetic conditions "
            "such as sickle cell anemia, cystic fibrosis, and specialized oncology treatments."
        )
    },
    {
        "id": "DOC13",
        "title": "The Renaissance: European Art, Science, and Humanism",
        "category": "History",
        "content": (
            "The European Renaissance witnessed a profound cultural and intellectual rebirth bridging the Middle Ages to modernity. "
            "Artists and polymaths like Leonardo da Vinci and Michelangelo revolutionized painting, anatomy, and engineering. "
            "The invention of the movable type printing press by Johannes Gutenberg democratized literacy, scholarship, and scientific inquiry."
        )
    },
    {
        "id": "DOC14",
        "title": "The Industrial Revolution: Steam Power, Mechanization, and Modern Society",
        "category": "History",
        "content": (
            "Originating in Britain during the eighteenth century, the Industrial Revolution transformed agrarian economies into "
            "mechanized industrial powerhouses. The steam engine, mechanized textile looms, iron smelting innovations, and locomotive "
            "railways stimulated urbanization, global trade, and modern economic structures."
        )
    },
    {
        "id": "DOC15",
        "title": "Quantum Computing: Qubits, Superposition, and Cryptography",
        "category": "Computer Science",
        "content": (
            "Quantum computers harness quantum mechanics phenomena like superposition and entanglement to process information. "
            "Unlike classical binary bits that exist as 0 or 1, quantum bits (qubits) can explore exponential computational paths "
            "simultaneously, presenting massive breakthroughs for molecular simulation and posing cryptographic challenges to RSA."
        )
    }
]
