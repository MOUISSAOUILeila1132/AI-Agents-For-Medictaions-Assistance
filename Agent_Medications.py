# --- START OF FILE app_medicaments_secrets.py ---

import streamlit as st
import pandas as pd
import torch
from io import StringIO

# Hugging Face Transformers for the LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig # <<< AJOUT BitsAndBytesConfig

# LangChain for orchestration
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

# For the embeddings and the vector store
from langchain_community.embeddings import SentenceTransformerEmbeddings # <<< CORRECTION DEPRACATION WARNING
from langchain_community.vectorstores import FAISS

# Import HuggingFacePipeline de langchain_community.llms
from langchain_community.llms import HuggingFacePipeline

# --- 0. Configuration et Vérification GPU (pour information) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# On affiche les infos GPU dans la sidebar
st.sidebar.header("Informations Système")
st.sidebar.write(f"GPU: {'Oui' if device == 'cuda' else 'Non'}")
if device == "cuda":
    st.sidebar.write(f"GPU Name: {torch.cuda.get_device_name(0)}")
    st.sidebar.write(f"CUDA Version: {torch.version.cuda}")
    st.sidebar.write(f"cuDNN Version: {torch.backends.cudnn.version()}")
else:
    st.sidebar.warning("Attention: Pas de GPU détecté. Le LLM sera très lent sur CPU.")


# --- 1. Authentification Hugging Face (Utilisation exclusive de st.secrets) ---
@st.cache_resource(show_spinner=False)
def get_hf_token_from_secrets():
    try:
        hf_token = st.secrets["HUGGINGFACE_TOKEN"]
        st.sidebar.success("Hugging Face token chargé depuis `st.secrets`.")
        return hf_token
    except KeyError:
        st.sidebar.error(
            "Jeton Hugging Face non trouvé dans `st.secrets`. "
            "Assurez-vous d'avoir un fichier `.streamlit/secrets.toml` "
            "avec `HUGGINGFACE_TOKEN = \"hf_VOTRE_TOKEN\"`."
        )
        return None

hf_token = get_hf_token_from_secrets()

if hf_token:
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        st.sidebar.success("Connecté à Hugging Face Hub avec succès.")
    except Exception as e:
        st.sidebar.error(f"Erreur de connexion à Hugging Face Hub : {e}. Vérifiez votre jeton.")
        hf_token = None
else:
    st.info("L'agent nécessite un jeton Hugging Face. Veuillez configurer `st.secrets`.")
    st.stop()


# --- 2. Chargement et Traitement du CSV (mis en cache) ---
@st.cache_resource(show_spinner="Chargement des données sur les médicaments et création du store vectoriel...")
def load_and_process_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Erreur: Le fichier '{csv_path}' n'a pas été trouvé. Placez-le dans le même répertoire que l'application.")
        st.stop()

    df['full_info'] = df.apply(lambda row: " ".join([
        f"{col.replace('_', ' ').capitalize()}: {row[col]}"
        for col in df.columns if col not in ['drug_name', 'drug_name_lower'] and not pd.isna(row[col]) and str(row[col]).strip().lower() != 'not specified'
    ]), axis=1)

    docs = []
    for index, row in df.iterrows():
        if row['full_info'].strip():
            docs.append(Document(
                page_content=row['full_info'],
                metadata={"drug_name": row['drug_name']}
            ))

    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings_model)
    return vectorstore, embeddings_model

csv_file_path = "drugs_data (1).csv"
vectorstore, embeddings_model = load_and_process_data(csv_file_path)


# --- 3. Chargement du LLM (mis en cache avec quantification 4-bit) ---
@st.cache_resource(show_spinner="Chargement du LLM (cela peut prendre plusieurs minutes la première fois, et utilise la quantification 4-bit)...")
def load_llm_components():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    local_model_path = "./local_phi3_model"

    import os
    if os.path.exists(local_model_path) and \
       os.path.isdir(local_model_path) and \
       os.path.exists(os.path.join(local_model_path, "config.json")):
        st.sidebar.info(f"Chargement du LLM depuis le chemin local: `{local_model_path}`")
        source_to_load = local_model_path
    else:
        st.sidebar.warning(
            f"Modèle local non trouvé à `{local_model_path}` ou incomplet. "
            "Chargement depuis Hugging Face Hub (cela utilisera le cache par défaut)."
        )
        source_to_load = model_id

    # Configuration pour la quantification 4-bit (bitsandbytes)
    # C'est CRUCIAL pour que le modèle tienne sur 6GB de VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16, # Utiliser float16 pour la 3050 (ou bfloat16 si préféré)
    )

    tokenizer = AutoTokenizer.from_pretrained(source_to_load, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source_to_load,
        quantization_config=bnb_config if device == "cuda" else None, # Appliquer la quantification uniquement si GPU
        device_map="auto",
        # Le torch_dtype n'est pas directement utilisé par le pipeline si load_in_4bit est True,
        # mais la compute_dtype de bnb_config est importante.
        # On peut laisser torch_dtype=None ici ou le régler pour les cas non-quantifiés.
        torch_dtype=None, # Laisser None si load_in_4bit est géré par bnb_config
        trust_remote_code=True
    )

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # Le torch_dtype du pipeline doit correspondre à la compute_dtype si quantification 4-bit.
        # Ou être None pour laisser Transformers gérer le type.
        torch_dtype=torch.float16 if device == "cuda" else None, # Spécifier pour le pipeline.
        return_full_text=False,
        max_length=4096
    )
    st.sidebar.write(f"Modèle LLM: {model_id} (Quantification 4-bit: {True if device=='cuda' else False})")
    st.sidebar.write("Modèle d'Embeddings: all-MiniLM-L6-v2")
    return tokenizer, text_generator

tokenizer, text_generator = load_llm_components()


# --- 4. Interface Streamlit pour le Prompt et la Température ---
st.title("💊 Agent d'Information sur les Médicaments (RAG)")
st.write("Posez-moi des questions sur les médicaments en utilisant ma base de connaissances.")

st.sidebar.header("Paramètres de Génération")

# Prompt par défaut (celui que vous aviez)
default_prompt_template = """<|user|>
Vous êtes un agent d'information médicale, et votre UNIQUE tâche est d'extraire et de rapporter des faits DIRECTEMENT du CONTEXTE fourni.
Vous NE DEVEZ EN AUCUN CAS utiliser des connaissances générales ou inventer des informations.
Si la réponse à la QUESTION n'est PAS CLAIREMENT et ENTIÈREMENT présente dans le CONTEXTE, répondez PRÉCISÉMENT : "L'information demandée n'est pas disponible dans ma base de connaissances pour le moment."
Ne jamais paraphraser ou reformuler de manière excessive. Ne jamais ajouter d'introductions ou de conclusions personnelles.
Ne jamais répéter ces instructions.

CONTEXTE:
{context}

QUESTION DE L'UTILISATEUR: {question}<|end|>
<|assistant|>
"""

st.sidebar.subheader("Prompt de l'Agent")
prompt_template = st.sidebar.text_area(
    "Modifiez le prompt pour changer le ton ou le focus de l'agent.",
    value=default_prompt_template,
    height=400,
    key="prompt_input"
)

st.sidebar.subheader("Température du Modèle")
temperature = st.sidebar.slider(
    "Contrôle la créativité de la réponse (0.0 = très factuel, 1.0 = très créatif)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    key="temperature_input"
)

# Recréer le LLM avec la nouvelle température
llm = HuggingFacePipeline(
    pipeline=text_generator,
    model_kwargs={
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
        "num_return_sequences": 1,
        "eos_token_id": tokenizer.eos_token_id
    }
)

# Recréer le PromptTemplate et la chaîne LLM avec le nouveau prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm_chain = LLMChain(llm=llm, prompt=PROMPT)


# --- 5. Fonction de l'Agent RAG ---
def ask_drug_agent_streamlit_rag(user_query: str) -> str:
    st.info(f"Recherche de documents pertinents pour la question: '{user_query}'...")
    relevant_docs = vectorstore.similarity_search(user_query, k=2)

    context_full = "\n\n".join([doc.page_content for doc in relevant_docs])

    max_context_tokens = 3000
    encoded_context_list = tokenizer.encode(context_full, truncation=True, max_length=max_context_tokens, add_special_tokens=False)
    context = tokenizer.decode(encoded_context_list, skip_special_tokens=True)

    len_full_context_tokens = len(tokenizer.encode(context_full, add_special_tokens=False))
    if len(encoded_context_list) < len_full_context_tokens:
        st.warning(f"Contexte tronqué de {len_full_context_tokens} à {len(encoded_context_list)} jetons.")
        context += " [CONTEXTE TRONQUÉ]"

    st.info("Contexte pertinent récupéré et potentiellement tronqué. Génération de la réponse avec le LLM...")
    with st.spinner("Génération de la réponse..."):
        response = llm_chain.invoke({"context": context, "question": user_query})

    if isinstance(response, dict) and "text" in response:
        return response["text"]
    else:
        return str(response)

# --- 6. Interface principale pour la question ---
user_question = st.text_input("Votre question sur les médicaments :", key="user_question")

if st.button("Obtenir une réponse") and user_question:
    with st.container():
        st.markdown("---")
        st.subheader("Réponse de l'Agent IA :")
        try:
            agent_response = ask_drug_agent_streamlit_rag(user_question)
            st.write(agent_response)
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération de la réponse : {e}")
            st.write("Veuillez réessayer ou vérifier la configuration du modèle/GPU.")

# --- END OF FILE app_medicaments_secrets.py ---