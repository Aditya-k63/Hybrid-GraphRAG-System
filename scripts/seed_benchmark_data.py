"""
Seed benchmark data into PostgreSQL for CI/CD testing.

Creates synthetic document chunks that match the benchmark questions
so the RAG system has data to search against.
"""

import json
import os
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent.parent / "benchmark_dataset.json"


def get_connection():
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "graphrag"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )
    register_vector(conn)
    return conn


TEST_CHUNKS = [
    # Transformer paper chunks
    {
        "source": "attention_is_all_you_need.pdf",
        "content": "The transformer is a new simple network architecture, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. The transformer was introduced in the paper 'Attention Is All You Need' by Vaswani et al. 2017.",
    },
    {
        "source": "attention_is_all_you_need.pdf",
        "content": "The authors of the transformer paper are Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. They were researchers at Google Brain, Google Research, and the University of Toronto.",
    },
    {
        "source": "attention_is_all_you_need.pdf",
        "content": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging otherwise inhibits this ability. The transformer uses scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.",
    },
    # NASA tech report chunks
    {
        "source": "nasa_tech_report.pdf",
        "content": "The Jet Propulsion Laboratory (JPL) is a federally funded research and development center managed by the California Institute of Technology (Caltech) for NASA. JPL is located in Pasadena, California. NASA owns the facility and funds its operations, while Caltech manages day-to-day operations and personnel.",
    },
    {
        "source": "nasa_tech_report.pdf",
        "content": "Key missions managed by JPL include the Voyager program (launched 1977) for outer planets exploration, the Mars Exploration Rovers Spirit and Opportunity (launched 2003), the Cassini-Huygens mission to Saturn (launched 1997), and the Juno mission to Jupiter (launched 2011).",
    },
    # Company annual report chunks
    {
        "source": "company_annual_report.pdf",
        "content": "The company reported revenue of $52.8 billion in fiscal year 2023, compared to $45.2 billion in 2022, representing a 16.8% year-over-year growth. Operating income grew 22% to $12.4 billion. Net income was $9.8 billion, up from $7.6 billion in the prior year.",
    },
    {
        "source": "company_annual_report.pdf",
        "content": "In Q3 2023, the company completed the acquisition of Startup X for $2.1 billion in cash. This acquisition added 150+ engineers and proprietary machine learning infrastructure to the cloud computing division. Cloud AI service revenue increased by 34% following the integration of Startup X's technology.",
    },
    # Medical paper chunks
    {
        "source": "medical_paper.pdf",
        "content": "CRISPR-Cas9 is a revolutionary gene editing technology that allows precise modification of DNA in living organisms. It uses a guide RNA (gRNA) to direct the Cas9 enzyme to a specific location in the genome, where it creates a double-strand break. The cell's natural repair mechanisms then modify the gene, enabling knockout, insertion, or correction of genetic sequences.",
    },
    {
        "source": "medical_paper.pdf",
        "content": "Ethical concerns surrounding gene editing include: germline editing which creates heritable changes affecting future generations, informed consent for experimental treatments, equitable access to expensive gene therapies, potential off-target effects causing unintended mutations, and the distinction between therapeutic applications (treating disease) versus enhancement applications (improving traits).",
    },
    # Wikipedia AI chunks
    {
        "source": "wikipedia_ai.pdf",
        "content": "Alan Turing is widely considered the father of artificial intelligence. His 1950 paper 'Computing Machinery and Intelligence' proposed the Turing Test as a measure of machine intelligence. Turing also conceptualized the universal Turing machine, which became the theoretical foundation for modern computers capable of running AI algorithms.",
    },
    {
        "source": "wikipedia_ai.pdf",
        "content": "Narrow AI (Artificial Narrow Intelligence or ANI) refers to AI systems designed and trained for a specific task, such as image recognition, language translation, or playing chess. General AI (Artificial General Intelligence or AGI) would possess human-level cognitive abilities across all domains, capable of learning and reasoning about any intellectual task that a human can perform.",
    },
    # Legal contract chunks
    {
        "source": "legal_contract.pdf",
        "content": "Termination clauses: This Agreement may be terminated (1) by mutual written consent of both parties, (2) with ninety (90) days written notice by either party, (3) immediately upon material breach that remains uncured for thirty (30) days after written notice of such breach, or (4) upon bankruptcy or insolvency of either party.",
    },
    {
        "source": "legal_contract.pdf",
        "content": "Section 8 (Limitation of Liability): The total aggregate liability of either party under this Agreement shall not exceed the total fees paid in the preceding twelve (12) months. Section 9 (Indemnification): Each party shall indemnify the other against third-party claims. The liability cap in Section 8 does not apply to indemnification obligations for IP infringement or gross negligence.",
    },
    # Cooking book chunk
    {
        "source": "cooking_book.pdf",
        "content": "The Maillard reaction is a chemical reaction between amino acids and reducing sugars that occurs when food is heated above 140C (280F). Named after French chemist Louis-Camille Maillard, this reaction is responsible for the brown color and complex flavors in seared meats, toasted bread, roasted coffee, and chocolate. Hundreds of different flavor compounds are created during the Maillard reaction.",
    },
    # History textbook chunk
    {
        "source": "history_textbook.pdf",
        "content": "The assassination of Archduke Franz Ferdinand of Austria-Hungary on June 28, 1914, in Sarajevo triggered a chain of events leading to World War I. Austria-Hungary issued an ultimatum to Serbia, which partially rejected it. Austria-Hungary declared war on Serbia, activating the alliance system: Russia mobilized to support Serbia, Germany declared war on Russia and France, France allied with Russia, and Britain entered the war when Germany invaded neutral Belgium.",
    },
]


def seed():
    print("Seeding benchmark test data...")
    conn = get_connection()
    cur = conn.cursor()

    # Check if data already exists
    cur.execute("SELECT COUNT(*) FROM document_sections")
    count = cur.fetchone()[0]
    if count > 0:
        print(f"Database already has {count} chunks. Skipping seed.")
        conn.close()
        return

    # Generate simple embeddings (random vectors for CI/CD testing)
    import numpy as np

    for i, chunk in enumerate(TEST_CHUNKS):
        embedding = np.random.randn(384).astype(float).tolist()
        metadata = {"source": chunk["source"], "chunk_index": i}
        cur.execute(
            "INSERT INTO document_sections (content, meta, embedding) VALUES (%s, %s, %s)",
            (chunk["content"], json.dumps(metadata), embedding),
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Seeded {len(TEST_CHUNKS)} chunks across {len(set(c['source'] for c in TEST_CHUNKS))} documents")


if __name__ == "__main__":
    seed()
