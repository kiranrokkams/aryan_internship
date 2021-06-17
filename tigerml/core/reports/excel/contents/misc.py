from ..helpers import write_title
from ...contents import Text


class ExcelText(Text):

	@classmethod
	def from_parent(cls, parent):
		return cls(parent.text)

	def save(self, worksheet, workbook, top_row, left_col, width=None):
		width = width or self.text_width
		if self.name:
			write_title(self.name, worksheet, workbook, top_row, left_col, width)
			top_row += 1
		if self.format:
			format = workbook.add_format(self.format)
		else:
			format = None
		if width > 1 or self.text_height > 1:
			worksheet.merge_range(top_row, left_col, top_row+self.text_height-1, left_col+width, self.text, cell_format=format)
		else:
			worksheet.write_string(top_row, left_col, self.text, cell_format=format)

