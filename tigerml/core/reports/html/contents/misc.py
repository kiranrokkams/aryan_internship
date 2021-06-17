from ...contents import Text, BaseContent


class HTMLText(Text):

	@classmethod
	def from_parent(cls, parent):
		return cls(parent.text)

	def to_html(self, resource_path=''):
		# html_str = ''
		# if self.name:
		# 	html_str += title_html(prettify_slug(self.name))
		text_html = '<p>{}</p>'.format(self.text)
		return '<div class="content text_content"><div class="content_inner">{}</div></div>'.format(text_html)

	def save(self):
		pass


class HTMLBase(BaseContent):

	def to_html(self, resource_path=''):
		return self.content

