EXPERIMENT_SET = [
    # first queries are from 5G_PFCP.
    # Category 1: Control questions:
    # RAG or no-RAG should not significantly affect the answer quality, as 
    # the answer should be based on general knowledge or reasoning rather 
    # than specific dataset context.
    {
        "id": "Q_PFCP_control_1", 
        "category": "control",
        "sample_id": "none",
        "xai_file": "none",
        "query": "What is the difference between supervised and unsupervised machine learning in the context of IDS?",
        "ground_truth": "Supervised uses labeled data (like 5GC_PFCP); unsupervised finds patterns in unlabeled data. This thesis uses supervised models (XGBoost, DNN)."
    },
    {
        "id": "Q_PFCP_control_2", 
        "category": "control",
        "sample_id": "none",
        "xai_file": "none",
        "query": "What does UPF stand for in 5G and what is its role in the N4 interface?",
        "ground_truth": "User Plane Function. It handles packet routing and is controlled by the SMF via PFCP on the N4 interface."
    },
    {
        "id": "Q_PFCP_control_3",
        "category": "control",
        "sample_id": "none",
        "xai_file": "none",
        "query": "Can SHAP values be negative? If a feature has a negative SHAP value in a binary attack/benign classification, does that mean the feature indicates the traffic is benign?",
        "ground_truth": "Yes, SHAP values can be negative. In binary classification where class 1 = Attack and class 0 = Benign, a negative SHAP value for a feature means that feature value pushes the prediction toward Benign (class 0), away from Attack. A positive SHAP value pushes toward Attack (class 1). The magnitude indicates the strength of the push. A feature can have a negative SHAP value even when its absolute feature value is high — what matters is whether that value is associated with attack or benign behavior in the model's learned representation. This is a control question testing basic XAI understanding that the LLM should answer correctly from training knowledge, without needing retrieval."
    },

    # Category 2: retrieval/context relevance,  
    # Testing if RAG finds the right docs.

    {
        "id": "Q_PFCP_retrieval_1", 
        "category": "retrieval",
        "sample_id": "2304",
        "xai_file": "evaluation_dataset/lime_individual_2304.txt",
        "query": "Sample 2304 shows Protocol=17 (UDP) and Port 8805. Based on available documentation, is the absence of TCP flags in this flow suspicious?",
        "ground_truth": "No. PFCP uses UDP. TCP flags (SYN/ACK) are irrelevant. RAG must retrieve PFCP/UDP specifications to verify this."
    },
    {
        "id": "Q_PFCP_retrieval_2", 
        "category": "retrieval",
        "sample_id": "none",
        "xai_file": "none",
        "query": "Source and destination ports are consistently 8805. What is the significance of this port in 5G, and is this a valid attack indicator?",
        "ground_truth": "Port 8805 is the standard IANA port for PFCP. It is expected for all samples and is not a suspicious indicator."
    },
    {
        "id": "Q_PFCP_retrieval_3", "category": "retrieval",
        "sample_id": "137",
        "xai_file": "evaluation_dataset/shap_individual_137.json",
        "query": "Sample 137, 5GC_PFCP dataset. PFCPSessionModificationRequest_counter=0 has SHAP value 0.149, pushing toward attack prediction. According to 5G PFCP documentation, is the absence of Session Modification requests during a 55-second flow suspicious, or is it normal for stable sessions to have no modification activity?",
        "ground_truth": "According to the 3GPP PFCP specification, Session Modification requests are only sent when session parameters need to change — for example when QoS rules are updated or bearer modifications occur. A stable PDU session with no changing requirements will naturally produce zero Session Modification requests. Therefore PFCPSessionModificationRequest_counter=0 over a 55-second window is entirely normal for a stable session and should not be treated as an attack indicator. The model is incorrectly penalizing normal stable session behavior, contributing to the false positive prediction for sample 137."
    },
    {
        "id": "Q_PFCP_retrieval_4", "category": "retrieval",
        "sample_id": "none",
        "xai_file": "none",
        "query": "In the 5GC_PFCP dataset, source port and destination port are both consistently 8805 across all samples. What is the significance of port 8805 in 5G networks, and should the consistent use of this port be considered a suspicious indicator in PFCP traffic analysis?",
        "ground_truth": "Port 8805 is the IANA-assigned standard port for PFCP (Packet Forwarding Control Protocol), defined in 3GPP TS 29.244. All PFCP signaling between the SMF and UPF on the N4 interface uses UDP port 8805. Therefore the consistent use of port 8805 in the 5GC_PFCP dataset is expected and not suspicious — it simply confirms the traffic is PFCP protocol traffic. Both attack and benign samples in this dataset use port 8805 because the dataset specifically captures PFCP traffic. A model that assigns high importance to port 8805 is not learning a security-relevant pattern. This is a control question where the LLM should answer correctly from training knowledge without needing retrieval from the knowledge base."
    },
     {
        "id": "Q_PFCP_retrieval_5", "category": "retrieval",
        "sample_id": "137",
        "xai_file": "evaluation_dataset/shap_individual_137.json",
        "query": "Sample 137, 5GC_PFCP dataset. PFCPHeartbeatRequest_counter=13 and PFCPHeartbeatResponse_counter=12 over a flow duration of 55 seconds. The request-response ratio is 13:12. According to PFCP documentation, what does a near-symmetric heartbeat request-response pattern indicate, and does one unanswered heartbeat constitute evidence of an attack?",
        "ground_truth": "A heartbeat request-response ratio of 13:12 indicates that almost all heartbeat requests received a response, which is characteristic of a healthy PFCP path management exchange. The single unanswered heartbeat (13 requests, 12 responses) is within normal network behavior — packet loss or timing issues can cause occasional missed responses without indicating an attack. According to the PFCP specification, path failure is only declared after multiple consecutive missed heartbeat responses. Therefore this pattern is consistent with normal keep-alive signaling and does not indicate a Heartbeat Flood attack. The model's attack prediction for this sample is a false positive."
    },
  
    # Category 3: Answer faithfulness to the retrieved explanations; 
    {
        "id": "Q_PFCP_faithfulness_1", 
        "category": "faithfulness",
        "sample_id": "137",
        "xai_file": "shap_individual_137.json",
        "query": "In Sample 137, which specific feature has the highest SHAP value, and how does its value (0.219) influence the prediction?",
        "ground_truth": "The 'Unnamed: 0' (row index) has the highest SHAP value (0.219), pushing the model toward an 'Attack' prediction."
    },
    {
        "id": "Q_PFCP_faithfulness_2",
        "category": "faithfulness",
        "sample_id": "250",
        "xai_file": "evaluation_dataset/shap_individual_250.json",
        "query": "Sample 250, 5GC_PFCP dataset TCP/IP layer, DNN, binary. Predicted: Attack (1), True: Attack (1). Top SHAP features: Src Port (SHAP=-0.208, pushes benign), Fwd Pkt Len Std=1.421 (SHAP=+0.046), PSH Flag Cnt=1.498 (SHAP=+0.045), Init Bwd Win Byts=1.450 (SHAP=+0.042), ACK Flag Cnt=1.489 (SHAP=+0.032). Src Port strongly pushes toward benign yet the model still predicts attack. What does the combination of high PSH flags, high ACK flags, and elevated Init Bwd Win Byts indicate about this traffic at the TCP/IP layer?",
        "ground_truth": "High PSH (Push) flag count indicates frequent data pushes to the application layer — in attack traffic this can indicate a session modification flood where the attacker sends many small messages flagged for immediate delivery. High ACK flag count combined with PSH suggests active bidirectional session manipulation. Init Bwd Win Byts=1.450 (normalized, indicating a large backward window size) suggests the receiving end advertised a large buffer, which attackers can exploit to send more data before receiving flow control signals. The negative Src Port SHAP value (-0.208) indicates the specific source port value pushed toward benign, but was outweighed by the combined attack signals from flag counts and window size. The DNN correctly identified this as an attack by weighting the combination of TCP-layer behavioral features over the port signal."
    },
    {
        "id": "Q_PFCP_faithfulness_3", "category": "faithfulness",
        "sample_id": "4045",
        "xai_file": "lime_individual_4045.txt",
        "query": "For Sample 4045, the top two LIME features push toward 'Benign'. Explain how the model still arrived at an 'Attack' prediction based on the LIME summary.",
        "ground_truth": "Model predictions are the sum of all weights plus a base value. Even if top features are benign, the collective weight of other features and the high base value (0.755) result in an 'Attack' verdict."
    },
    {
        "id": "Q_PFCP_faithfulness_4", "category": "faithfulness",
        "sample_id": "4045",
        "xai_file": "evaluation_dataset/lime_individual_4045.txt",
        "query": "Sample 4045, 5GC_PFCP TCP/IP layer, MLP, binary. Predicted: Attack (1), True: Attack (1). Feature values are normalized (z-scores). Top LIME features: Src Port>0.94 (-0.518, pushes benign), Down/Up Ratio>-0.06 (-0.240, pushes benign), Pkt Len Std>0.07 (+0.139, pushes attack). Two of the top three LIME features push toward benign yet the model predicts attack. Is a prediction valid when the majority of top features push toward the opposite class?",
        "ground_truth": "A prediction can be valid even when the majority of top-ranked features push toward the opposite class, because SHAP and LIME importance rankings are not votes — they are magnitude-weighted contributions. The correct interpretation is to sum all contributions: if the features pushing toward attack collectively outweigh those pushing toward benign, the prediction is attack regardless of how many features point each way. In this case, while Src Port and Down/Up Ratio push toward benign, their combined magnitude (-0.518 + -0.240 = -0.758) must be compared against all features pushing toward attack. Additionally, Pkt Len Std, Idle Std, Bwd Seg Size Avg, and Fwd Pkt Len Std all push toward attack. The base value of the model already starts at 0.755 (strongly toward attack), meaning the default prior without any features is already attack-leaning for this dataset. This is a semantically important interpretation point that a RAG-grounded explanation should communicate clearly."
    },
    {
        "id": "Q_GAD_faithfulness_1", "category": "faithfulness",
        "sample_id": "95",
        "xai_file": "evaluation_dataset/lime_individual_95.json",
        "query":"tcp_flags contributes positively toward attack, while most payload-related features push toward benign. What does this conflict suggest about the traffic pattern?"
    },
    {   # SHAP
        "id":"Q_NIDD_faith_1",
        "sample_id": "228693",
        "xai_file":"evaluation_dataset/shap_individual_228693.json",
        "query": "For sample 228693, which feature has the highest SHAP value, and what does it indicate about the prediction?"
    },
    {   # SHAP
        "id":"Q_NIDD_faith_2",
        "sample_id": "228693",
        "xai_file":"evaluation_dataset/shap_individual_228693.json",
        "query": "In sample 228693, both ‘Seq’ and ‘sTtl’ have high positive SHAP values. How do they jointly influence the attack prediction?"
    },
    {
        # LIME 
        "id": "Q_NIDD_faith_3",
        "sample_id": "69166",
        "xai_file": "evaluation_dataset/lime_individual_69166.txt",
        "query":"In sample 69166, some features push toward benign (e.g. sMeanPktSz), yet the prediction is attack. Why?"
    },
    

    #  Category 4:Answer relevance/usefulness.
    {
        "id": "Q_PFCP_usefulness_1", "category": "usefulness",
        "sample_id": "668",
        "xai_file": "shap_individual_668.json",
        "query": "Sample 668 shows a Flow IAT Min of 4μs but a Max of 11s. Does this timing pattern align with a specific PFCP attack type described in the documentation?",
        "ground_truth": "Yes. This 'burst-then-idle' pattern is characteristic of a PFCP Flood attack, where messages are sent in rapid succession to exhaust resources."
    },
    {
        "id": "Q_PFCP_usefulness_2", "category": "usefulness",
        "sample_id": "8",
        "xai_file": "evaluation_dataset/shap_individual_8.json",
        "query": "Sample 8, 5GC_PFCP dataset. PFCPHeartbeatRequest_counter=13 over a 55-second flow. According to the PFCP dataset documentation, what is the definition of a PFCP Session Establishment Flood attack, and does a heartbeat count of 13 per minute meet the threshold for flood attack classification?",
        "ground_truth": "According to the 5GC_PFCP dataset documentation, a PFCP Session Establishment Flood Attack aims to exhaust UPF resources by sending excessive Session Establishment Requests and Heartbeat Requests. The attack is characterized by randomized Session IDs and high message volume targeting resource exhaustion. A heartbeat count of 13 over 55 seconds corresponds to approximately one heartbeat every 4.2 seconds, which is within the typical PFCP keepalive interval range. This rate does not meet the threshold for a flood attack, which would involve orders-of-magnitude higher message rates. The true flood attack samples in the dataset would show dramatically higher counter values. Sample 8 should be classified as benign based on heartbeat rate alone."
    },
    {
        "id": "Q_PFCP_usefulness_3", "category": "usefulness",
        "sample_id": "81",
        "query": "The feature 'Unnamed: 0' appears as a top importance feature. Based on data science principles, should this be trusted for a real-world 5G deployment?",
        "ground_truth": "No. This indicates a 'spurious correlation' or data leakage from the dataset's row ordering. The RAG should flag this as a model reliability issue."
    },
    {
        "id": "Q_PFCP_usefulness_4", "category": "usefulness",
        "sample_id": "137_and_8",
        "xai_file": "none",
        "query": "In the 5GC_PFCP dataset, both sample 137 (LightGBM) and sample 8 (DecisionTree) show PFCPHeartbeatRequest_counter=13 and PFCPSessionModificationRequest_counter=0 over ~55 seconds. LightGBM predicted Attack (false positive) while DecisionTree predicted Benign (false negative). What does this disagreement between models indicate about the reliability of these features for PFCP attack detection?",
        "ground_truth": "The same feature values producing opposite predictions across two models reveals that PFCPHeartbeatRequest_counter=13 and PFCPSessionModificationRequest_counter=0 over ~55 seconds are ambiguous features that fall near the decision boundary. Neither pattern is unambiguously malicious or benign according to the PFCP specification — 13 heartbeats per minute is within normal operating range, and zero session modifications is normal for stable sessions. The inter-model disagreement indicates that neither model has learned a robust rule for this pattern, and that the underlying feature space does not cleanly separate this traffic type. A RAG-grounded explanation should communicate this ambiguity to the security analyst rather than presenting a confident attack or benign verdict."
    },
    {
        "id": "Q_PFCP_usefulness_5", "category": "usefulness",
        "sample_id": "668_and_4844",
        "xai_file": "none",
        "query": "Compare sample 668 (XGBoost, attack, Fwd IAT Min=53μs, Flow Pkts/s=0.73) and sample 4844 (DecisionTree, benign, Fwd IAT Min=25409μs, Flow Pkts/s=9.24). Sample 4844 has 12x higher packet rate but was correctly classified as benign, while sample 668 was correctly classified as attack. According to PFCP documentation, why is inter-arrival time a more reliable attack indicator than packet rate for PFCP flood detection?",
        "ground_truth": "In PFCP traffic, inter-arrival time is a more reliable attack indicator than raw packet rate because PFCP flood attacks are characterized by bursting behavior — sending many messages in rapid succession — rather than sustained high throughput. A minimum IAT of 53 microseconds means packets are arriving near-simultaneously in bursts, which is physically impossible in normal PFCP signaling where messages follow request-response patterns with processing delays. In contrast, a sustained packet rate of 9.24 packets/second with minimum IAT of 25ms is consistent with regular application traffic. The PFCP dataset documentation notes that flood attacks aim to exhaust UPF resources through high message volume — this volume manifests as burst behavior detectable through minimum IAT, not average packet rate. This explains why timing variance features dominate the SHAP explanations for attack samples while packet rate features have lower importance."
    },
    {
        "id": "Q_PFCP_usefulness_6", "category": "usefulness",
        "sample_id": "81",
        "xai_file": "evaluation_dataset/lime_individual_81.txt",
        "query": "Sample 81, 5GC_PFCP dataset, XGBoost, binary. Predicted: Attack (1), True: Attack (1). Top LIME features: duration<=55007224 (-0.273), PFCPSessionModificationRequest_counter<=0 (+0.184), Unnamed:0>905.5 (+0.172). The row index Unnamed:0 appears again as a top-3 feature. Across multiple samples in this dataset, Unnamed:0 appears as a high-importance feature. What does this suggest about the model's reliability for deployment in a real 5G network?",
        "ground_truth": "The repeated appearance of Unnamed:0 (dataset row index) as a high-importance feature across multiple samples indicates that the model has learned a spurious correlation between row position and class label. This is a data quality artifact — in the original dataset, benign and malicious samples are likely grouped or ordered such that row position correlates with class. A row index carries zero network security semantics and would not exist in a real deployment scenario. This finding indicates the model would likely perform significantly worse on real production traffic than its evaluation metrics suggest. Any security explanation that includes Unnamed:0 as a justification for an attack prediction should be flagged as unreliable. This is a critical finding for the RAG layer to identify and communicate."
    },
    {
        "id": "Q_PFCP_usefulness_7", "category": "usefulness",
        "sample_id": "2141",
        "xai_file": "evaluation_dataset/shap_individual_2141.json",
        "query": "Sample 2141, 5GC_PFCP dataset, RandomForest, binary. Predicted: Attack (1), True: Attack (1). Top SHAP features: Flow IAT Std=4158057 (SHAP=0.048), Bwd IAT Max=11001599μs (SHAP=0.041), Active Max=364μs (SHAP=0.041), Bwd IAT Std=5500759 (SHAP=0.041). No single feature dominates — the prediction is driven by many small contributions from IAT variance features. What does a prediction based on many small timing variance signals rather than one dominant feature indicate about the attack pattern?",
        "ground_truth": "A prediction driven by many small IAT variance contributions rather than a single dominant feature suggests the model has detected a distributed timing anomaly rather than a single obvious indicator. This is consistent with a PFCP flood attack that uses irregular timing to evade simple threshold-based detection. The high Bwd IAT Max (~11 seconds) and high Bwd IAT Std indicate the backward flow has highly irregular timing, which is not characteristic of normal PFCP response patterns where responses closely follow requests. Active Mean=364μs (the time the connection was actively transmitting) being very short relative to the overall flow duration also indicates brief bursts of activity. This type of multi-signal low-magnitude prediction is actually harder to explain to a security analyst than a single dominant feature, which highlights the value of RAG-grounded explanations that can synthesize multiple weak signals into a coherent security narrative."
    },
    {
        "id": "Q_PFCP_usefulness_8", "category": "usefulness",
        "sample_id": "4844",
        "xai_file": "evaluation_dataset/shap_individual_4844.json",
        "query": "Sample 4844, 5GC_PFCP dataset, DecisionTree, binary. Predicted: Benign (0), True: Benign (0). Top SHAP features: Fwd IAT Min=25409μs (SHAP=-0.211, pushes benign), Flow IAT Std=19496 (SHAP=-0.161), Active Mean=0 (SHAP=-0.042). Flow Pkts/s=9.24 is relatively high. Despite high packet rate, the model correctly classified this as benign. Why does a high packet rate not indicate an attack here, and what features drove the benign classification?",
        "ground_truth": "A packet rate of 9.24 packets/second is not inherently malicious in PFCP traffic. The benign classification is driven by the regular timing pattern: Fwd IAT Min=25409 microseconds (~25ms) indicates the minimum gap between packets is substantial, meaning no burst behavior exists. The low Flow IAT Std=19496 relative to the mean indicates consistent, regular inter-arrival times — characteristic of legitimate application traffic with steady throughput. Active Mean=0 indicates no active transmission periods were detected, consistent with a session that transmits in regular intervals without burst activity. This sample correctly illustrates that packet rate alone is insufficient to classify PFCP traffic as malicious — the timing regularity is the key discriminating factor. This is a correct model decision that the RAG explanation should reinforce."
    },
    {
        "id": "Q_PFCP_usefulness_9", "category": "usefulness",
        "sample_id": "8",
        "xai_file": "evaluation_dataset/shap_individual_8.json",
        "query": "Sample 8, 5GC_PFCP dataset, DecisionTree, binary. Predicted: Benign (0), True: Malicious (1) — false negative. Top SHAP features: PFCPHeartbeatRequest_counter=13 (SHAP=+1.385, pushes toward attack), Unnamed:0=722 (SHAP=+1.136), duration=55009728 (SHAP=-0.290, pushes toward benign), PFCPSessionModificationRequest_counter=0 (SHAP=+0.329). The model missed this attack despite the heartbeat counter being the strongest feature. Why did the duration feature override the heartbeat signal?",
        "ground_truth": "The flow duration of ~55 seconds (55009728 microseconds) pushed strongly toward benign (SHAP=-0.290) because the model learned that legitimate PFCP heartbeat sessions typically last around 55 seconds — this is consistent with the standard heartbeat interval. The model therefore interpreted a 55-second duration as evidence of normal keep-alive behavior, which outweighed the suspicious heartbeat count signal. This is a false negative caused by the model conflating normal session duration with benign intent. In reality, an attacker can maintain a 55-second window deliberately to mimic normal timing while still flooding heartbeats. The RAG-grounded explanation should identify that duration alone is insufficient to classify a flow as benign when heartbeat counts are elevated."
    },
    {
        "id": "Q_5GAD_usefulness_1", "category": "usefulness",
        "sample_id": "7915",
        "xai_file": "evaluation_dataset/shap_individual_7915.json",
        "query": "Payload_std (33.36) has the highest SHAP value (+0.149). What does high payload variance indicate in network traffic, and why might it be associated with malicious behavior?"
    },
    {
        "id": "Q_5GAD_usefulness_2", "category": "usefulness",
        "sample_id": "7915",
        "xai_file": "evaluation_dataset/shap_individual_7915.json",
        "query": "The prediction is driven by payload_std, payload_mean, and payload_max. What does the combination of high payload variability and relatively large payload sizes suggest about this traffic pattern?"
    },
    {
        "id": "Q_5GAD_usefulness_3", "category": "usefulness",
        "sample_id": "7915",
        "xai_file": "evaluation_dataset/shap_individual_7915.json",
        "query": "The sample shows tcp_flags=2 and udp_len=0. What does this indicate about the transport protocol, and is this consistent with the rest of the features?"
    },
    {
        "id": "Q_NIDD_use_1",
        "category": "usefulness",
        "sample_id": "228693",
        "xai_file": "evaluation_dataset/shap_individual_228693.json",
        "query": "In sample 228693, DstPkts=0 and DstBytes=0. What does this indicate about the communication pattern, and is it suspicious?"
    },
    {
        "id": "Q_NIDD_use_2",
        "category": "usefulness",
        "sample_id": "69166",
        "xai_file": "evaluation_dataset/shap_individual_69166.json",
        "query": "Sample 69166 shows active TCP features (SynAck, AckDat, TcpRtt). What kind of network behavior does this suggest?",

    }
]
