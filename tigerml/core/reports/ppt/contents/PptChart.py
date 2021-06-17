from ...contents import Chart


class PptChart(Chart):

	@classmethod
	def from_parent(cls, parent):
		return cls(parent.plot)

	def save(self, placeholder, slide_obj):
		pass

