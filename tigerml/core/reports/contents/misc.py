
class Text:

	def __init__(self, text, name='', width=1, height=1, format={}):
		self.text = text
		self.name = name
		self.text_width = int(width)
		self.text_height = height
		self.format = format

	@property
	def width(self):
		return self.text_width

	@property
	def height(self):
		return self.text_height + (1 if self.name else 0)


class BaseContent:

	def __init__(self, content, name=''):
		self.content = content
		self.name = name

	@property
	def width(self):
		return None

	@property
	def height(self):
		return None

