class FaceDescriptor:
     def __init__(self):
        self.features_map =  {
            "nose": ["nose_size_shape", "nose_ridge", "nose_width", "nose_tip_angle", "nose_tip_size_shape", "nostrils_size_shape"],
            "eyes": ["eyes_spacing", "eyes_angle", "eyes_depth", "eye_puffs", "eyelashes", "eye_color", "eyes_corner_indents_and_eyes_iris_size", "eyelids_top", "eyelids_bottom"],
            "eyebrows": ["eyebrows_basic_shapes", "eyebrows_position", "eyebrows_color"],
            "forehead": ["forehead_shapes", "forehead_lines"],
            "mouth": ["mouth_size", "mouth_angle", "lips_size", "teeth", "smiles"],
            "jaw_chin": ["cheeks", "jaws", "chins", "dimples", "clefts"],
            "ears": ["ears_size", "ears_cups_ridges", "ears_placement", "ears_height"],
            "face_overview": ["face_shape", "face_type", "head_type", "face_color", "ear_eyebrow_combinations", "chin_eyebrow_combinations", "face_lines", "facial_hair"]
        
                }
    def get_prompt(self, face_part:str):
        if face_part not in self.features_map:
            raise ValueError(f"Unknown face part: {part_name}")
        
        features = self.features_map[part_name]
        feature_list = "\n".join([f"- {feature}" for feature in features])

        return f""" 
        You are a facial morphology analyzer.
        Analyze ONLY the {face_part} visible in the image.
        Describe the following features: {feature_list}
        Rules:
        1. Use ONLY visual observations.
        2. Do NOT infer personality.
        3. Do NOT infer emotions.
        4. Do NOT infer character traits.
        5. If a feature cannot be determined, return null.
        6. Return VALID JSON ONLY.
        Output format:{{"feature_name": {{"value": "...", "confidence": 0.0, "description": "..."}}}}
        The JSON keys MUST exactly match the feature names listed above.
        """

    def describe_part(self, part_name, part_img, features):
        prompt = self.get_prompt(part_name, features)
        #part_img for vll 
