class FaceDescriptor:
    self.features_map = {
        "nose": ["nose_size_shape", "nose_ridge", "nose_width", "nose_tip_angle", "nose_tip_size_shape", "nostrils_size_shape"]
        
    }
    def get_prompt(self, face_part, face_part_description):
        """
        Where face_part is string 
        face_part_description is list of string 
        """
        return f"You are a facial morphology analyzer, Analyze ONLY the {face_part}. Descripe the following features if it were avaliable {face_part_description}. Use only visual observations. 
        Do not infer personality. Return valid JSON only."

    def describe_part(self, part_name, part_img, features):
        prompt = self.get_prompt(part_name, features)
        #part_img for vll 
        
    def describe_nose(self, image_path):
        features = 
        prompt = self.get_prompt("nose", features)
        
    def describe_eyes(self, image_path):
        features = ["eyes_spacing", "eyes_angle", "eyes_depth", "eye_puffs", "eyelashes", "eye_color", "eyes_corner_indents_and_eyes_iris_size", "eyelids_top", "eyelids_bottom"]
        prompt = self.get_prompt("mouth", features)

    def describe_eyebrows(self, image_path):
        features = ["eyebrows_basic_shapes", "eyebrows_position", "eyebrows_color"]
        prompt = self.get_prompt("eyebrows", features)

    def describe_forehead(self, image_path):
        features = ["forehead_shapes", "forehead_lines"]
        prompt = self.get_prompt("forehead", features)

    def describe_mouth(self, image_path):
        features = ["mouth_size", "mouth_angle", "lips_size", "teeth", "smiles"]
        prompt = self.get_prompt("mouth", features)

    def describe_ears(self, image_path):
        features = ["ears_size", "ears_cups_ridges", "ears_placement", "ears_height"]
        prompt = self.get_prompt("ears", features)

    def describe_jaw_chin(self, image_path):
        features = ["cheeks", "jaws", "chins", "dimples", "clefts"]
        prompt = self.get_prompt("jaw_chin", features)

    def describe_whole_face(self, image_path):
        features = ["face_shape", "face_type", "head_type", "face_color", "ear_eyebrow_combinations", "chin_eyebrow_combinations", "face_lines", "facial_hair"]
        prompt = self.get_prompt("face", features)
