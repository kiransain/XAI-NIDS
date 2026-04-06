EXPERIMENT_SET = [
    {
    "id":"Q1",
    "sample_id": "137",
    "xai_file": "evaluation_dataset/shap_individual_137.json",
    "query": "Explain why this PFCP flow is classified as an attack using the given SHAP values. Focus on the top contributing features and relate them to potential network attack behavior."
},
    # {   # this is from 5GC_PFCP\binary\DNN
    #     "id": "Q1",
    #     "sample_id": "1",
    #     "xai_file": "evaluation_dataset\shap_individual_250.json",
    #     "query": "Explain why this PFCP flow is classified as an attack using the given SHAP values. Focus on the top contributing features and relate them to potential network attack behavior.",
    #     "ground_truth": "Model answer"
    # },
    # Generic question
    {
        "id": "Q2",
        "sample_id": "2",
        "xai_file": "none",
        "query": "Are DoS attacks even a threat for 5G-applications?",
        # "retrieved_context": " The network or a node will eventually undergo resource exhaustion and deny the nonmalicious users’ access as a result of a DoS attack. Due to the heterogeneous nature of the 5G networks, DoS attacks impose a vital threat in 5G which may target the network nodes, devices, and applications [54]. This work focuses on the DoS/DDoS attacks targeted at user devices, but the attack may congest the network too.",
        # "answer": "Yes, DoS (Denial of Service) attacks are indeed a significant threat to 5G applications. The increased connectivity and reliance on 5G networks for critical applications (such as IoT, autonomous vehicles, and smart cities) make them attractive targets for attackers. DoS attacks can disrupt the availability of services, degrade performance, and potentially lead to financial losses or safety risks. Therefore, it is crucial to implement robust security measures to protect 5G applications from such threats.",
        "ground_truth": "DoS attacks impose a vital threat in 5G which may target the network nodes, devices, and applications"
        },
    # 5G PFCP / binary / DEcision Tree ; true positive case
    {
    "id": "Q3",
    "sample_id": "2304",
    "xai_file": "evaluation_dataset/lime_individual_2304.txt",
    "query": "Sample 2304, 5GC_PFCP dataset, LightGBM, binary classification. Predicted: Attack (1), True: Attack (1). Top LIME features: Fwd Seg Size Avg<=16 (+0.156), Bwd IAT Std>5500600 (+0.107), Flow IAT Mean>1571699 (+0.081), Src Port<=8805 (+0.073). All packet sizes are exactly 16 bytes and no TCP flags are set. What does uniform packet size with zero TCP flags indicate in PFCP traffic, and is this pattern consistent with a known PFCP attack type?",
    "ground_truth": "In PFCP traffic, all packets being exactly 16 bytes with zero TCP flags (no SYN, ACK, PSH, FIN, RST) indicates pure UDP-based PFCP signaling with minimal payload. Port 8805 is the standard PFCP port. The uniform 16-byte packet size is consistent with PFCP Heartbeat messages, which carry minimal payload. The high Bwd IAT Std and Flow IAT Mean indicate irregular inter-arrival timing, which combined with the Src Port=8805 pattern, is consistent with a PFCP Session Establishment Flood or Heartbeat flood attack where an attacker sends repeated minimal PFCP messages at irregular intervals to exhaust UPF resources. The model correctly identified this as malicious."
    },
    {
    "id": "Q4",
    "sample_id": "2304",
    "xai_file": "evaluation_dataset/lime_individual_2304.txt",
    "query": "Sample 2304, 5GC_PFCP dataset. The flow shows Src Port=8805, Dst Port=8805, Protocol=17 (UDP), zero PSH flags, zero SYN flags, zero ACK flags, and Down/Up Ratio=1.0. Based on 5G PFCP documentation, what type of PFCP message exchange does this traffic pattern represent, and should the absence of TCP flags in PFCP traffic be considered suspicious?",
    "ground_truth": "PFCP operates over UDP port 8805 by definition — the absence of TCP flags is entirely expected and not suspicious on its own. TCP flags are irrelevant to PFCP since it uses UDP. A Down/Up Ratio of 1.0 indicates symmetric bidirectional traffic, consistent with request-response pairs typical of PFCP Heartbeat exchanges. The model should not treat zero TCP flags as an attack indicator in PFCP context. This query tests whether the RAG can correctly ground the explanation in PFCP protocol semantics to avoid misinterpretation."
    },
    # 5G PFCP / binary / DEcision Tree ; False positive case
    {
    "id": "Q5",
    "sample_id": "137",
    "xai_file": "evaluation_dataset/shap_individual_137.json",
    "query": "Sample 137, 5GC_PFCP dataset, LightGBM. Predicted: Attack (1), True: Benign (0) — false positive. SHAP top features: Unnamed:0 row index (SHAP=0.219), PFCPSessionModificationRequest_counter=0 (SHAP=0.149), PFCPHeartbeatRequest_counter=13 (SHAP=0.056). The row index feature has the highest SHAP value. Is a dataset row index a meaningful security feature, and what does this indicate about model reliability?",
    "ground_truth": "A dataset row index (Unnamed:0) should never be a meaningful security feature — it carries no network semantics and its high SHAP value (0.219) indicates the model has learned a spurious correlation with the row position in the dataset. This is a data quality issue rather than a true security signal. The model's false positive prediction is therefore unreliable. A correct security explanation should dismiss the row index contribution and focus only on semantically meaningful PFCP features. This highlights a limitation of the underlying IDS model that the RAG-grounded explanation should ideally identify and flag."
},
# ground_truths generated using LLMs. (e.g. GPT 5.3)
{
    "id": "Q6",
    "sample_id": "137",
    "xai_file": "evaluation_dataset/shap_individual_137.json",
    "query": "Sample 137, 5GC_PFCP dataset. PFCPSessionModificationRequest_counter=0 has SHAP value 0.149, pushing toward attack prediction. According to 5G PFCP documentation, is the absence of Session Modification requests during a 55-second flow suspicious, or is it normal for stable sessions to have no modification activity?",
    "ground_truth": "According to the 3GPP PFCP specification, Session Modification requests are only sent when session parameters need to change — for example when QoS rules are updated or bearer modifications occur. A stable PDU session with no changing requirements will naturally produce zero Session Modification requests. Therefore PFCPSessionModificationRequest_counter=0 over a 55-second window is entirely normal for a stable session and should not be treated as an attack indicator. The model is incorrectly penalizing normal stable session behavior, contributing to the false positive prediction for sample 137."
},
{
    "id": "Q7",
    "sample_id": "137",
    "xai_file": "evaluation_dataset/shap_individual_137.json",
    "query": "Sample 137, 5GC_PFCP dataset. PFCPHeartbeatRequest_counter=13 and PFCPHeartbeatResponse_counter=12 over a flow duration of 55 seconds. The request-response ratio is 13:12. According to PFCP documentation, what does a near-symmetric heartbeat request-response pattern indicate, and does one unanswered heartbeat constitute evidence of an attack?",
    "ground_truth": "A heartbeat request-response ratio of 13:12 indicates that almost all heartbeat requests received a response, which is characteristic of a healthy PFCP path management exchange. The single unanswered heartbeat (13 requests, 12 responses) is within normal network behavior — packet loss or timing issues can cause occasional missed responses without indicating an attack. According to the PFCP specification, path failure is only declared after multiple consecutive missed heartbeat responses. Therefore this pattern is consistent with normal keep-alive signaling and does not indicate a Heartbeat Flood attack. The model's attack prediction for this sample is a false positive."
},
{
    "id": "Q8",
    "sample_id": "8",
    "xai_file": "evaluation_dataset/shap_individual_8.json",
    "query": "Sample 8, 5GC_PFCP dataset, DecisionTree, binary. Predicted: Benign (0), True: Malicious (1) — false negative. Top SHAP features: PFCPHeartbeatRequest_counter=13 (SHAP=+1.385, pushes toward attack), Unnamed:0=722 (SHAP=+1.136), duration=55009728 (SHAP=-0.290, pushes toward benign), PFCPSessionModificationRequest_counter=0 (SHAP=+0.329). The model missed this attack despite the heartbeat counter being the strongest feature. Why did the duration feature override the heartbeat signal?",
    "ground_truth": "The flow duration of ~55 seconds (55009728 microseconds) pushed strongly toward benign (SHAP=-0.290) because the model learned that legitimate PFCP heartbeat sessions typically last around 55 seconds — this is consistent with the standard heartbeat interval. The model therefore interpreted a 55-second duration as evidence of normal keep-alive behavior, which outweighed the suspicious heartbeat count signal. This is a false negative caused by the model conflating normal session duration with benign intent. In reality, an attacker can maintain a 55-second window deliberately to mimic normal timing while still flooding heartbeats. The RAG-grounded explanation should identify that duration alone is insufficient to classify a flow as benign when heartbeat counts are elevated."
},{
    "id": "Q9",
    "sample_id": "81",
    "xai_file": "evaluation_dataset/lime_individual_81.txt",
    "query": "Sample 81, 5GC_PFCP dataset, XGBoost, binary. Predicted: Attack (1), True: Attack (1). Top LIME features: duration<=55007224 (-0.273), PFCPSessionModificationRequest_counter<=0 (+0.184), Unnamed:0>905.5 (+0.172). The row index Unnamed:0 appears again as a top-3 feature. Across multiple samples in this dataset, Unnamed:0 appears as a high-importance feature. What does this suggest about the model's reliability for deployment in a real 5G network?",
    "ground_truth": "The repeated appearance of Unnamed:0 (dataset row index) as a high-importance feature across multiple samples indicates that the model has learned a spurious correlation between row position and class label. This is a data quality artifact — in the original dataset, benign and malicious samples are likely grouped or ordered such that row position correlates with class. A row index carries zero network security semantics and would not exist in a real deployment scenario. This finding indicates the model would likely perform significantly worse on real production traffic than its evaluation metrics suggest. Any security explanation that includes Unnamed:0 as a justification for an attack prediction should be flagged as unreliable. This is a critical finding for the RAG layer to identify and communicate."
},{
    "id": "Q10",
    "sample_id": "none",
    "xai_file": "none",
    "query": "What is the difference between supervised and unsupervised machine learning? Which approach is used in intrusion detection systems?",
    "ground_truth": "Supervised learning trains models on labeled data where each sample has a known output class. Unsupervised learning finds patterns in unlabeled data. Intrusion detection systems typically use supervised learning when labeled attack datasets are available, as in this thesis which uses labeled datasets (5GC_PFCP, 5G-NIDD, 5GAD) with binary and multiclass labels. The models evaluated include LightGBM, XGBoost, RandomForest, DecisionTree, MLP, and DNN — all supervised classifiers. This is a control question where the knowledge base should not provide relevant context, testing whether the LLM performs equally with or without retrieval on general ML concepts."
},{
    "id": "Q11",
    "sample_id": "none",
    "xai_file": "none",
    "query": "What does UPF stand for in 5G networks and what is its primary function?",
    "ground_truth": "UPF stands for User Plane Function. It is a core component of the 5G Core network responsible for packet routing and forwarding, traffic usage reporting, QoS handling, and acting as the anchor point for mobility between different access technologies. The UPF connects the 5G radio access network to external data networks. In the context of PFCP, the UPF is controlled by the SMF (Session Management Function) via the N4 interface using PFCP messages. This is a control question — the answer is well-known general 5G knowledge that the LLM should answer correctly without retrieval. RAG should not significantly improve the response."
},{
    "id": "Q12",
    "sample_id": "137_and_8",
    "xai_file": "none",
    "query": "In the 5GC_PFCP dataset, both sample 137 (LightGBM) and sample 8 (DecisionTree) show PFCPHeartbeatRequest_counter=13 and PFCPSessionModificationRequest_counter=0 over ~55 seconds. LightGBM predicted Attack (false positive) while DecisionTree predicted Benign (false negative). What does this disagreement between models indicate about the reliability of these features for PFCP attack detection?",
    "ground_truth": "The same feature values producing opposite predictions across two models reveals that PFCPHeartbeatRequest_counter=13 and PFCPSessionModificationRequest_counter=0 over ~55 seconds are ambiguous features that fall near the decision boundary. Neither pattern is unambiguously malicious or benign according to the PFCP specification — 13 heartbeats per minute is within normal operating range, and zero session modifications is normal for stable sessions. The inter-model disagreement indicates that neither model has learned a robust rule for this pattern, and that the underlying feature space does not cleanly separate this traffic type. A RAG-grounded explanation should communicate this ambiguity to the security analyst rather than presenting a confident attack or benign verdict."
},{
    "id": "Q13",
    "sample_id": "8",
    "xai_file": "evaluation_dataset/shap_individual_8.json",
    "query": "Sample 8, 5GC_PFCP dataset. PFCPHeartbeatRequest_counter=13 over a 55-second flow. According to the PFCP dataset documentation, what is the definition of a PFCP Session Establishment Flood attack, and does a heartbeat count of 13 per minute meet the threshold for flood attack classification?",
    "ground_truth": "According to the 5GC_PFCP dataset documentation, a PFCP Session Establishment Flood Attack aims to exhaust UPF resources by sending excessive Session Establishment Requests and Heartbeat Requests. The attack is characterized by randomized Session IDs and high message volume targeting resource exhaustion. A heartbeat count of 13 over 55 seconds corresponds to approximately one heartbeat every 4.2 seconds, which is within the typical PFCP keepalive interval range. This rate does not meet the threshold for a flood attack, which would involve orders-of-magnitude higher message rates. The true flood attack samples in the dataset would show dramatically higher counter values. Sample 8 should be classified as benign based on heartbeat rate alone."
},
{
    "id": "Q14",
    "sample_id": "none",
    "xai_file": "none",
    "query": "Can SHAP values be negative? If a feature has a negative SHAP value in a binary attack/benign classification, does that mean the feature indicates the traffic is benign?",
    "ground_truth": "Yes, SHAP values can be negative. In binary classification where class 1 = Attack and class 0 = Benign, a negative SHAP value for a feature means that feature value pushes the prediction toward Benign (class 0), away from Attack. A positive SHAP value pushes toward Attack (class 1). The magnitude indicates the strength of the push. A feature can have a negative SHAP value even when its absolute feature value is high — what matters is whether that value is associated with attack or benign behavior in the model's learned representation. This is a control question testing basic XAI understanding that the LLM should answer correctly from training knowledge, without needing retrieval."
},
{
    "id": "Q15",
    "sample_id": "668",
    "xai_file": "evaluation_dataset/shap_individual_668.json",
    "query": "Sample 668, 5GC_PFCP dataset, XGBoost, binary. Predicted: Attack (1), True: Attack (1). Top SHAP features: Fwd IAT Min=53μs (SHAP=2.513), Flow IAT Min=4μs (SHAP=1.724), Fwd IAT Max=11001620μs (SHAP=1.722), Fwd IAT Std=7779282 (SHAP=1.304). The minimum inter-arrival time is 4 microseconds while the maximum is ~11 seconds. What does this extreme spread between minimum and maximum IAT values indicate about the nature of this traffic in a PFCP context?",
    "ground_truth": "An extremely small minimum inter-arrival time (4 microseconds) combined with a very large maximum IAT (~11 seconds) indicates a burst-then-idle traffic pattern. In PFCP context, legitimate heartbeat traffic has regular, evenly-spaced inter-arrival times of approximately 5 seconds. A 4-microsecond minimum IAT indicates packets being sent in rapid succession — far faster than any legitimate PFCP signaling. This burst pattern is characteristic of PFCP flood attacks where an attacker sends many messages in rapid bursts interspersed with idle periods to mimic session-level timing while generating abnormally high instantaneous message rates. The high Fwd IAT Std confirms this irregularity. The model correctly identified this as an attack based on timing anomalies rather than message count alone."
},
{
    "id": "Q16",
    "sample_id": "2141",
    "xai_file": "evaluation_dataset/shap_individual_2141.json",
    "query": "Sample 2141, 5GC_PFCP dataset, RandomForest, binary. Predicted: Attack (1), True: Attack (1). Top SHAP features: Flow IAT Std=4158057 (SHAP=0.048), Bwd IAT Max=11001599μs (SHAP=0.041), Active Max=364μs (SHAP=0.041), Bwd IAT Std=5500759 (SHAP=0.041). No single feature dominates — the prediction is driven by many small contributions from IAT variance features. What does a prediction based on many small timing variance signals rather than one dominant feature indicate about the attack pattern?",
    "ground_truth": "A prediction driven by many small IAT variance contributions rather than a single dominant feature suggests the model has detected a distributed timing anomaly rather than a single obvious indicator. This is consistent with a PFCP flood attack that uses irregular timing to evade simple threshold-based detection. The high Bwd IAT Max (~11 seconds) and high Bwd IAT Std indicate the backward flow has highly irregular timing, which is not characteristic of normal PFCP response patterns where responses closely follow requests. Active Mean=364μs (the time the connection was actively transmitting) being very short relative to the overall flow duration also indicates brief bursts of activity. This type of multi-signal low-magnitude prediction is actually harder to explain to a security analyst than a single dominant feature, which highlights the value of RAG-grounded explanations that can synthesize multiple weak signals into a coherent security narrative."
},
{
    "id": "Q17",
    "sample_id": "4844",
    "xai_file": "evaluation_dataset/shap_individual_4844.json",
    "query": "Sample 4844, 5GC_PFCP dataset, DecisionTree, binary. Predicted: Benign (0), True: Benign (0). Top SHAP features: Fwd IAT Min=25409μs (SHAP=-0.211, pushes benign), Flow IAT Std=19496 (SHAP=-0.161), Active Mean=0 (SHAP=-0.042). Flow Pkts/s=9.24 is relatively high. Despite high packet rate, the model correctly classified this as benign. Why does a high packet rate not indicate an attack here, and what features drove the benign classification?",
    "ground_truth": "A packet rate of 9.24 packets/second is not inherently malicious in PFCP traffic. The benign classification is driven by the regular timing pattern: Fwd IAT Min=25409 microseconds (~25ms) indicates the minimum gap between packets is substantial, meaning no burst behavior exists. The low Flow IAT Std=19496 relative to the mean indicates consistent, regular inter-arrival times — characteristic of legitimate application traffic with steady throughput. Active Mean=0 indicates no active transmission periods were detected, consistent with a session that transmits in regular intervals without burst activity. This sample correctly illustrates that packet rate alone is insufficient to classify PFCP traffic as malicious — the timing regularity is the key discriminating factor. This is a correct model decision that the RAG explanation should reinforce."
},
{
    "id": "Q18",
    "sample_id": "668_and_4844",
    "xai_file": "none",
    "query": "Compare sample 668 (XGBoost, attack, Fwd IAT Min=53μs, Flow Pkts/s=0.73) and sample 4844 (DecisionTree, benign, Fwd IAT Min=25409μs, Flow Pkts/s=9.24). Sample 4844 has 12x higher packet rate but was correctly classified as benign, while sample 668 was correctly classified as attack. According to PFCP documentation, why is inter-arrival time a more reliable attack indicator than packet rate for PFCP flood detection?",
    "ground_truth": "In PFCP traffic, inter-arrival time is a more reliable attack indicator than raw packet rate because PFCP flood attacks are characterized by bursting behavior — sending many messages in rapid succession — rather than sustained high throughput. A minimum IAT of 53 microseconds means packets are arriving near-simultaneously in bursts, which is physically impossible in normal PFCP signaling where messages follow request-response patterns with processing delays. In contrast, a sustained packet rate of 9.24 packets/second with minimum IAT of 25ms is consistent with regular application traffic. The PFCP dataset documentation notes that flood attacks aim to exhaust UPF resources through high message volume — this volume manifests as burst behavior detectable through minimum IAT, not average packet rate. This explains why timing variance features dominate the SHAP explanations for attack samples while packet rate features have lower importance."
},
{
    "id": "Q19",
    "sample_id": "none",
    "xai_file": "none",
    "query": "In the 5GC_PFCP dataset, source port and destination port are both consistently 8805 across all samples. What is the significance of port 8805 in 5G networks, and should the consistent use of this port be considered a suspicious indicator in PFCP traffic analysis?",
    "ground_truth": "Port 8805 is the IANA-assigned standard port for PFCP (Packet Forwarding Control Protocol), defined in 3GPP TS 29.244. All PFCP signaling between the SMF and UPF on the N4 interface uses UDP port 8805. Therefore the consistent use of port 8805 in the 5GC_PFCP dataset is expected and not suspicious — it simply confirms the traffic is PFCP protocol traffic. Both attack and benign samples in this dataset use port 8805 because the dataset specifically captures PFCP traffic. A model that assigns high importance to port 8805 is not learning a security-relevant pattern. This is a control question where the LLM should answer correctly from training knowledge without needing retrieval from the knowledge base."
},
{
    "id": "Q20",
    "sample_id": "none",
    "xai_file": "none",
    "query": "In this thesis, both SHAP and LIME are used to explain IDS model predictions. What is the fundamental difference between SHAP and LIME as explanation methods, and in what scenario might they produce conflicting feature importance rankings for the same prediction?",
    "ground_truth": "SHAP (SHapley Additive exPlanations) uses game theory to compute exact feature contributions by considering all possible feature subsets, providing globally consistent explanations with mathematical guarantees. LIME (Local Interpretable Model-agnostic Explanations) approximates the model locally around a specific instance by perturbing the input and fitting a simple linear model to the perturbations. They can produce conflicting rankings when: (1) the decision boundary is highly nonlinear near the sample — LIME's local linear approximation may miss interactions that SHAP captures; (2) feature correlations exist — SHAP distributes credit across correlated features while LIME may assign importance arbitrarily between them; (3) for high-dimensional PFCP data with many zero-valued counters, LIME's perturbation strategy may generate unrealistic samples that distort the local approximation. This is a control question testing XAI background knowledge that does not require knowledge base retrieval."
},
{
    "id": "Q21",
    "sample_id": "250",
    "xai_file": "evaluation_dataset/shap_individual_250.json",
    "query": "Sample 250, 5GC_PFCP dataset TCP/IP layer, DNN, binary. Predicted: Attack (1), True: Attack (1). Top SHAP features: Src Port (SHAP=-0.208, pushes benign), Fwd Pkt Len Std=1.421 (SHAP=+0.046), PSH Flag Cnt=1.498 (SHAP=+0.045), Init Bwd Win Byts=1.450 (SHAP=+0.042), ACK Flag Cnt=1.489 (SHAP=+0.032). Src Port strongly pushes toward benign yet the model still predicts attack. What does the combination of high PSH flags, high ACK flags, and elevated Init Bwd Win Byts indicate about this traffic at the TCP/IP layer?",
    "ground_truth": "High PSH (Push) flag count indicates frequent data pushes to the application layer — in attack traffic this can indicate a session modification flood where the attacker sends many small messages flagged for immediate delivery. High ACK flag count combined with PSH suggests active bidirectional session manipulation. Init Bwd Win Byts=1.450 (normalized, indicating a large backward window size) suggests the receiving end advertised a large buffer, which attackers can exploit to send more data before receiving flow control signals. The negative Src Port SHAP value (-0.208) indicates the specific source port value pushed toward benign, but was outweighed by the combined attack signals from flag counts and window size. The DNN correctly identified this as an attack by weighting the combination of TCP-layer behavioral features over the port signal."
},
{
    "id": "Q22",
    "sample_id": "4045",
    "xai_file": "evaluation_dataset/lime_individual_4045.txt",
    "query": "Sample 4045, 5GC_PFCP TCP/IP layer, MLP, binary. Predicted: Attack (1), True: Attack (1). Feature values are normalized (z-scores). Top LIME features: Src Port>0.94 (-0.518, pushes benign), Down/Up Ratio>-0.06 (-0.240, pushes benign), Pkt Len Std>0.07 (+0.139, pushes attack). Two of the top three LIME features push toward benign yet the model predicts attack. Is a prediction valid when the majority of top features push toward the opposite class?",
    "ground_truth": "A prediction can be valid even when the majority of top-ranked features push toward the opposite class, because SHAP and LIME importance rankings are not votes — they are magnitude-weighted contributions. The correct interpretation is to sum all contributions: if the features pushing toward attack collectively outweigh those pushing toward benign, the prediction is attack regardless of how many features point each way. In this case, while Src Port and Down/Up Ratio push toward benign, their combined magnitude (-0.518 + -0.240 = -0.758) must be compared against all features pushing toward attack. Additionally, Pkt Len Std, Idle Std, Bwd Seg Size Avg, and Fwd Pkt Len Std all push toward attack. The base value of the model already starts at 0.755 (strongly toward attack), meaning the default prior without any features is already attack-leaning for this dataset. This is a semantically important interpretation point that a RAG-grounded explanation should communicate clearly."
},

#chatgpt generated questions:
{
  "id": "QU1",
  "sample_id": "none",
  "xai_file": "none",
  "query": "If a feature has the highest SHAP value in a prediction, does that mean the feature caused the attack, or only that the model relied on that feature? Explain the difference between feature importance and causality in the context of intrusion detection.",
  "ground_truth": "A high SHAP value does not mean the feature caused the attack. SHAP values explain the model's decision, not real-world causality. A feature with high SHAP value means the model relied heavily on that feature for its prediction. However, the feature may only be correlated with attacks in the dataset rather than being the true cause of malicious behavior. In intrusion detection, this distinction is important because models may learn spurious correlations, such as dataset artifacts or timing patterns, that are not actual attack causes. Therefore SHAP explains model behavior, not attacker behavior."
},
{
  "id": "QU2",
  "sample_id": "none",
  "xai_file": "none",
  "query": "What is the difference between BM25 retrieval and vector similarity search in a RAG system, and which one is better for retrieving technical documentation?",
  "ground_truth": "BM25 is a lexical retrieval method based on keyword matching and term frequency, while vector similarity search uses semantic embeddings to retrieve text with similar meaning rather than exact words. BM25 performs well when queries contain exact technical terms that also appear in documents. Vector search performs better when the query uses different wording than the document but has similar meaning. For technical documentation, vector search often performs better because it can match concepts and explanations even when exact keywords differ. However, BM25 can outperform vector search for very specific protocol names or exact terminology."
},
{
  "id": "QU3",
  "sample_id": "none",
  "xai_file": "none",
  "query": "Why can Retrieval-Augmented Generation (RAG) improve explanations of SHAP values for intrusion detection systems compared to using an LLM without retrieval?",
  "ground_truth": "SHAP values only indicate which features influenced the model prediction, but they do not explain the domain meaning of those features. An LLM without retrieval may generate generic explanations that are not specific to the protocol or dataset. RAG allows the model to retrieve technical documentation, dataset descriptions, and protocol specifications, which enables explanations that connect feature importance to actual network behavior. Therefore RAG improves explanations by grounding the reasoning in domain knowledge rather than relying only on the model's internal knowledge."
},
{
  "id": "QU4",
  "sample_id": "none",
  "xai_file": "none",
  "query": "If a dataset row index (Unnamed:0) appears as an important feature in SHAP explanations, could this indicate data leakage? Explain why or why not.",
  "ground_truth": "Yes, this can indicate data leakage or dataset artifacts. A row index should not contain any information about network traffic behavior. If the model uses the row index as a predictive feature, it means that the ordering of samples in the dataset is correlated with the labels. This can happen if benign and malicious samples are grouped or sorted in the dataset. The model then learns the sample position instead of real traffic characteristics. This is not true data leakage in the sense of using future information, but it is a dataset artifact that leads to unrealistic model performance and poor generalization."
},
{
  "id": "QU5",
  "sample_id": "none",
  "xai_file": "none",
  "query": "Why are documents split into chunks before being embedded and stored in a vector database in a RAG pipeline?",
  "ground_truth": "Documents are split into chunks because embedding models have input length limits and because retrieval works better on smaller, focused text segments. If entire documents were embedded as a single vector, the embedding would represent too many topics and retrieval would be less precise. Chunking allows the retriever to find specific relevant passages rather than entire documents, improving the quality of the retrieved context and therefore the quality of the generated answers."
},
{
  "id": "QU6",
  "sample_id": "none",
  "xai_file": "none",
  "query": "In intrusion detection systems, which is usually more critical: false positives or false negatives? Explain why.",
  "ground_truth": "Both false positives and false negatives are problematic, but false negatives are usually more critical because they represent real attacks that the system fails to detect. However, a very high false positive rate can make a system unusable because security analysts will ignore alerts. In practice, intrusion detection systems must balance both, but many systems prioritize reducing false negatives to avoid missed attacks while keeping false positives at a manageable level."
}
]
