from ...contents import Text


class PptText(Text):

	@classmethod
	def from_parent(cls, parent):
		return cls(parent.text)

	def save(self, placeholder, slide_obj):
		placeholder.text = self.text

