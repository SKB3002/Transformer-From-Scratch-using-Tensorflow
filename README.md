# Transformer-From-Scratch-using-Tensorflow
This project is a **complete, from-scratch implementation of the original Transformer architecture**  from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
# Transformer From Scratch (TensorFlow)

## 🔹 Highlights
- Implemented **Encoder, Decoder, Multi-Head Attention, and Positional Encoding** step by step
- Visualized **Self-Attention Heatmaps**
- Built **without using high-level seq2seq APIs**, to deeply understand core mechanics

## 📂 Project Structure
- `transformer_scratch.ipynb` – Full step-by-step implementation
- `src` - Code broken down into smallc hunks for better understanding
- `images/` – Sample attention maps and architecture diagrams
- `requirements.txt` – Required dependencies

## 🔹 Example Outputs
Example: **Self-Attention in the Encoder**  
- Sentence: `"I made transformer from scratch"`
- Observe how each word attends to others in the sequence:
- ![alt text]((https://github.com/SKB3002/Transformer-From-Scratch-using-Tensorflow/blob/main/Images/Heatmap%20of%20attention%20weights.png)


## 🔹 Key Learnings
- Why **Scaled Dot-Product Attention** is essential
- Role of **Multi-Head Attention** in capturing context
- How **Positional Encoding** allows sequence understanding without RNNs

## 📜 Reference
- Vaswani et al., 2017: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

---

⭐ **Next Step:** I will apply this architecture to a **Hinglish → English/Hindi Translator** as a real-world NLP project.
