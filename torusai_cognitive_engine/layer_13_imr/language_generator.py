# From User Part 1 - EnhancedLanguageGenerator
# Maps to README Layer 13: Introspective Meta-Reporter (IMR)
# Note: Uses EnhancedConceptGraph. Relative import will be needed.
# from ..layer_04_dcg.concept_graph import EnhancedConceptGraph # Example of potential import

class EnhancedLanguageGenerator:
    def __init__(self, cg_instance, narrative_log_list): # cg is EnhancedConceptGraph, narrative_log_list is a list
        self.cg = cg_instance
        self.narrative = narrative_log_list 
        self.templates = {
            "question_what": "What does {subject} {verb_base}?",
            "statement": "{subject} {verb_present} {object}."
        }

    def _conjugate_present(self, verb_base, is_plural_subject): # Renamed to avoid conflict if made static/utility
        if verb_base in ["be"]: return "are" if is_plural_subject else "is"
        if verb_base in ["have"]: return "have" if is_plural_subject else "has"
        if is_plural_subject: return verb_base
        # Simple rule, might need expansion for irregular verbs or different tenses
        return verb_base + "s" if not verb_base.endswith(("s", "sh", "ch", "x", "z")) else verb_base + "es"

    def _get_concept_word(self, cid, role, is_plural_subject=False): # Renamed
        node_data = self.cg.nodes.get(cid)
        if not node_data:
            # print(f"Warning: Node {cid} not found in concept graph for language generation.")
            return f"[{cid}_unknown]" # Fallback for missing node ID

        word_forms = node_data.get("wordForms", {"base": str(cid)})
        base_form = word_forms.get("base", str(cid))
        
        if role == "subject": 
            # Basic pluralization for subjects, may need enhancement
            return base_form + "s" if is_plural_subject and not base_form.endswith("s") else base_form
        if role == "object": 
            return base_form
        if role == "verb_base": 
            return base_form
        if role == "verb_present": 
            return self._conjugate_present(base_form, is_plural_subject)
        return base_form

    def generate(self, query_string):
        active_concepts = self.cg.getActiveConceptPath() # Method from EnhancedConceptGraph
        
        # Default CIDs if not enough active concepts
        subj_cid = active_concepts[0]["id"] if len(active_concepts) > 0 else "default_subject"
        verb_cid = active_concepts[1]["id"] if len(active_concepts) > 1 else "default_verb"
        obj_cid = active_concepts[2]["id"] if len(active_concepts) > 2 else "default_object"

        # Ensure default concepts exist in the graph for generation, or handle their absence
        # This is a simplistic way; a real system might have robust default concept handling
        if subj_cid == "default_subject" and subj_cid not in self.cg.nodes: self.cg.addEnhancedNode(subj_cid, linguistics={"wordForms": {"base":"thing"}})
        if verb_cid == "default_verb" and verb_cid not in self.cg.nodes: self.cg.addEnhancedNode(verb_cid, linguistics={"wordForms": {"base":"do"}})
        if obj_cid == "default_object" and obj_cid not in self.cg.nodes: self.cg.addEnhancedNode(obj_cid, linguistics={"wordForms": {"base":"something"}})
        
        # For this basic generator, assume subject is singular unless logic is added
        is_plural_subject = False 

        template_key = "question_what" if "?" in query_string else "statement"
        chosen_template = self.templates[template_key]
        
        generated_sentence = chosen_template.format(
            subject=self._get_concept_word(subj_cid, "subject", is_plural_subject),
            verb_base=self._get_concept_word(verb_cid, "verb_base"), # for questions like "What does X {verb_base}?"
            verb_present=self._get_concept_word(verb_cid, "verb_present", is_plural_subject), # for statements
            object=self._get_concept_word(obj_cid, "object")
        ).strip()

        self.narrative.append(generated_sentence)
        return generated_sentence
