
# this file contains the experiment set for the benchmark evaluation.

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
        "ground_truth":"Yes, SHAP values can be negative. In binary classification (Attack vs Benign), a negative SHAP value indicates that the feature contributes toward the benign class, while a positive value contributes toward the attack class. The magnitude reflects the strength of the contribution. This question tests basic interpretability of additive feature attribution methods in XAI."    },

    # Category 2: retrieval/context relevance,  
    # Testing if RAG finds the right docs.

    {
        "id": "Q_PFCP_retrieval_1", 
        "category": "retrieval",
        "sample_id": "2304",
        "xai_file": "evaluation_dataset/lime_individual_2304.txt",
        "query": "Sample 2304 shows Protocol=17 (UDP) and Port 8805. Based on available documentation, is the absence of TCP flags in this flow suspicious?",
        "ground_truth":"PFCP runs over UDP on port 8805, so TCP flags such as SYN/ACK are not applicable. The absence of TCP flags is expected and not suspicious in PFCP traffic."
    },
    {
        "id": "Q_PFCP_retrieval_3", "category": "retrieval",
        "sample_id": "137",
        "xai_file": "evaluation_dataset/shap_individual_137.json",
        "query": "Sample 137, 5GC_PFCP dataset. PFCPSessionModificationRequest_counter=0 has SHAP value 0.149, pushing toward attack prediction. According to 5G PFCP documentation, is the absence of Session Modification requests during a 55-second flow suspicious, or is it normal for stable sessions to have no modification activity?",
        "ground_truth": "According to PFCP/3GPP TS 29.244, Session Modification Requests are only generated when session parameters change (e.g., QoS updates or policy modifications). Therefore, a value of zero over a stable 55-second flow is expected in normal operation. This pattern alone is not sufficient to indicate malicious behavior and must be interpreted in context with other signaling features."
    },
    {
        "id": "Q_PFCP_retrieval_5", "category": "retrieval",
        "sample_id": "137",
        "xai_file": "evaluation_dataset/shap_individual_137.json",
        "query": "Sample 137, 5GC_PFCP dataset. PFCPHeartbeatRequest_counter=13 and PFCPHeartbeatResponse_counter=12 over a flow duration of 55 seconds. The request-response ratio is 13:12. According to PFCP documentation, what does a near-symmetric heartbeat request-response pattern indicate, and does one unanswered heartbeat constitute evidence of an attack?",
        "ground_truth":"A near-balanced heartbeat request-response ratio (13:12) is consistent with normal PFCP keep-alive behavior, where occasional missed responses can occur due to network delay or packet loss. According to PFCP specifications, a single missed response is not sufficient to indicate a path failure or attack condition. Only repeated consecutive failures would indicate abnormal behavior. This pattern is therefore consistent with benign PFCP signaling."
    },
  
    # Category 3: Answer faithfulness to the retrieved explanations; 
    {
        "id": "Q_PFCP_faithfulness_1", 
        "category": "faithfulness",
        "sample_id": "137",
        "xai_file": "evaluation_dataset/shap_individual_137.json",
        "query": "In Sample 137, which specific feature has the highest SHAP value, and how does its value (0.219) influence the prediction?",
        "ground_truth": "The feature 'Unnamed: 0' has the highest SHAP value (~+0.219), contributing toward the attack prediction. This indicates the model is relying on a dataset artifact (row index) rather than a meaningful network feature, suggesting potential data leakage."    },
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
        "xai_file": "evaluation_dataset/lime_individual_4045.txt",
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
        "xai_file": "evaluation_dataset/lime_individual_95.txt",
        "query":"tcp_flags contributes positively toward attack, while most payload-related features push toward benign. What does this conflict suggest about the traffic pattern?",
        "ground_truth": "High tcp_flags suggest active session manipulation, while payload-related features pushing toward benign suggest the attack does not carry a malicious payload. This is characteristic of a TCP-level flood (like SYN/ACK flood) where the handshake is the attack, not the data."
    },
    {   # SHAP
        "id":"Q_NIDD_faith_1", "category": "faithfulness",
        "sample_id": "228693",
        "xai_file":"evaluation_dataset/shap_individual_228693.json",
        "query": "For sample 228693, which feature has the highest SHAP value, and what does it indicate about the prediction?",
        "ground_truth":"The feature 'Seq' (Sequence number) has the highest SHAP value. This indicates the model is heavily weighting the packet ordering or sequence gaps as the primary indicator of an attack."
    },
    {   # SHAP
        "id":"Q_NIDD_faith_2", "category": "faithfulness",
        "sample_id": "228693",
        "xai_file":"evaluation_dataset/shap_individual_228693.json",
        "query": "In sample 228693, both ‘Seq’ and ‘sTtl’ have high positive SHAP values. How do they jointly influence the attack prediction?",
        "ground_truth": "Both 'Seq' (Sequence number) and 'sTtl' (Source TTL) having high positive SHAP values means that the model is interpreting specific sequence patterns and TTL values as strong indicators of attack behavior. The combination suggests that the model has learned to associate certain packet ordering anomalies (captured by 'Seq') and unusual TTL values (captured by 'sTtl') with malicious activity in NIDD traffic. This joint influence indicates that the attack prediction is not based on a single feature but rather on a pattern of features that together signal suspicious behavior."
    },
    {
        # LIME 
        "id": "Q_NIDD_faith_3", "category": "faithfulness",
        "sample_id": "69166",
        "xai_file": "evaluation_dataset/lime_individual_69166.txt",
        "query":"In sample 69166, some features push toward benign (e.g. sMeanPktSz), yet the prediction is attack. Why?",
        "ground_truth":"The attack prediction is driven by 'Seq' and 'sTtl' magnitude, which outweighs the benign signal from 'sMeanPktSz'. LIME shows the model prioritizes network-layer anomalies over application-layer packet sizes."
    },
    

    # #  Category 4:Answer relevance/usefulness.
    {
        "id": "Q_PFCP_usefulness_1", "category": "usefulness",
        "sample_id": "668",
        "xai_file": "evaluation_dataset/shap_individual_668.json",
        "query": "Sample 668 shows a Flow IAT Min of 4μs but a Max of 11s. Does this timing pattern align with a specific PFCP attack type described in the documentation?",
        "ground_truth": "Yes. This 'burst-then-idle' pattern is characteristic of a PFCP Flood attack, where messages are sent in rapid succession to exhaust resources."
    },
    {
        "id": "Q_PFCP_usefulness_2", "category": "usefulness",
        "sample_id": "8",
        "xai_file": "evaluation_dataset/shap_individual_8.json",
        "query": "Sample 8, 5GC_PFCP dataset. PFCPHeartbeatRequest_counter=13 over a 55-second flow. According to the PFCP dataset documentation, what is the definition of a PFCP Session Establishment Flood attack, and does a heartbeat count of 13 per minute meet the threshold for flood attack classification?",
        "ground_truth": "According to the 5GC_PFCP dataset description, PFCP flood attacks are characterized by unusually high rates of Session Establishment and Heartbeat messages aimed at exhausting UPF resources. A heartbeat rate of 13 over 55 seconds corresponds to a low-frequency keepalive pattern typical of normal PFCP operation. This rate is not indicative of flooding behavior when considered in isolation and must be evaluated alongside other signaling and timing features."
        },
    {
        "id": "Q_PFCP_usefulness_3", "category": "usefulness",
        "sample_id": "8",
        "xai_file": "evaluation_dataset/shap_individual_8.json",
        "query": "The feature 'Unnamed: 0' appears as a top importance feature. Based on data science principles, should this be trusted for a real-world 5G deployment?",
        "ground_truth": "No. 'Unnamed: 0' is the row index from the source CSV file. If a model relies on this, it has learned a spurious correlation (data leakage) based on the order of samples in the dataset rather than actual network behavior. It would fail in a real-world deployment where row indices do not exist."
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
        "xai_file": "evaluation_dataset/shap_individual_668.json", # using 668's SHAP file for retrieval since it contains the relevant features
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
        "query": "Sample 8, 5GC_PFCP dataset, DecisionTree, binary. Predicted: Benign (0), True: Malicious (1) — false negative. Top SHAP features: PFCPHeartbeatRequest_counter=13 (SHAP=+1.385, pushes toward attack), Unnamed:0=722 (SHAP=+1.136), duration=55009728 (SHAP=-0.290, pushes toward benign), PFCPSessionModificationRequest_counter=0 (SHAP=+0.329). The model missed this attack despite the heartbeat counter being the strongest feature. Why did the model predict benign despite strong positive SHAP contributions toward attack?",
        "ground_truth": "The model misclassifies this sample as benign, resulting in a false negative. The prediction is based on a combination of competing feature contributions rather than a single dominant factor. The strongest positive contributions toward the attack class come from PFCPHeartbeatRequest_counter (SHAP = +1.385) and Unnamed: 0 (SHAP = +1.136), followed by a smaller positive contribution from PFCPSessionModificationRequest_counter (SHAP = +0.329). These are counterbalanced by negative contributions, most notably duration (SHAP = -0.290), along with smaller negative effects from other features such as packet-level counters. The final prediction reflects the aggregated effect of both positive and negative feature contributions, resulting in a decision boundary tilt toward the benign class. The presence of a strong contribution from Unnamed: 0 suggests that the model may also be influenced by non-semantic or dataset-specific artifacts, which could affect interpretability and robustness."
    },
    {
        "id": "Q_5GAD_usefulness_1", "category": "usefulness",
        "sample_id": "7915",
        "xai_file": "evaluation_dataset/shap_individual_7915.json",
        "query": "Payload_std (33.36) has the highest SHAP value (+0.149). What does high payload variance indicate in network traffic, and why might it be associated with malicious behavior?",
        "ground_truth":"High payload variance (std) suggests irregular data sizes in a flow. In 5G, this can indicate a protocol exploitation where variable-length malformed packets are used to test UPF buffer vulnerabilities."
    },
    {
        "id": "Q_5GAD_usefulness_2", "category": "usefulness",
        "sample_id": "7915",
        "xai_file": "evaluation_dataset/shap_individual_7915.json",
        "query": "The prediction is driven by payload_std, payload_mean, and payload_max. What does the combination of high payload variability and relatively large payload sizes suggest about this traffic pattern?",
        "ground_truth": "This combination suggests a 'Heavy Hitter' or Data Exfiltration attempt, where large, variable payloads are being moved, deviating from the steady-state small packets of standard PFCP signaling."
    },
    {
        "id": "Q_5GAD_usefulness_3", "category": "usefulness",
        "sample_id": "7915",
        "xai_file": "evaluation_dataset/shap_individual_7915.json",
        "query": "The sample shows tcp_flags=2 and udp_len=0. What does this indicate about the transport protocol, and is this consistent with the rest of the features?",
        "ground_truth": "tcp_flags=2 (SYN) with udp_len=0 is inconsistent if the traffic is labeled as UDP. However, if the 5GAD dataset treats 'payload' generically, it indicates a SYN flood attempt where no UDP data is present because the protocol is actually TCP-based."
    },
    {
        "id": "Q_NIDD_use_1",
        "category": "usefulness",
        "sample_id": "228693",
        "xai_file": "evaluation_dataset/shap_individual_228693.json",
        "query": "In sample 228693, DstPkts=0 and DstBytes=0. What does this indicate about the communication pattern, and is it suspicious?",
        "ground_truth": "DstPkts=0 and DstBytes=0 indicate a unidirectional flow where no response was observed from the destination. This can occur in cases such as blocked traffic, dropped packets, or scanning behavior. While it may be consistent with reconnaissance or half-open connection attempts, it is not sufficient alone to definitively classify the behavior as malicious without additional context."
    
    },
    {
        "id": "Q_NIDD_use_2",
        "category": "usefulness",
        "sample_id": "69166",
        "xai_file": "evaluation_dataset/lime_individual_69166.txt",
        "query": "Sample 69166 shows active TCP features (SynAck, AckDat, TcpRtt). What kind of network behavior does this suggest?",
        "ground_truth": "The presence of SynAck and TcpRtt suggests a full TCP handshake was completed and the connection reached an established state. This indicates the traffic is not a simple SYN-only flood, but involves active bidirectional communication where round-trip latency can be measured."
    },
]
