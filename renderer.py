from PIL import Image, ImageDraw, ImageFont
import textwrap

class MangaRenderer:
    def __init__(self, font_path="C:/Windows/Fonts/arial.ttf"):
        self.font_path = font_path

    def get_font_size_that_fits(self, text, draw, box_width, box_height, max_font_size=100):
        low = 5
        high = max_font_size
        best_size = low
        
        while low <= high:
            mid = (low + high) // 2
            font = ImageFont.truetype(self.font_path, mid)
            avg_char_width = sum(draw.textlength(c, font=font) for c in "abcdefghijklmnopqrstuvwxyz")/26
            chars_per_line = max(1, int(box_width / avg_char_width))
            wrapped_text = textwrap.fill(text, width=chars_per_line)
            
            # Calculate bounding box of wrapped text
            bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            if w <= box_width and h <= box_height:
                best_size = mid
                low = mid + 1
            else:
                high = mid - 1
                
        return best_size

    def render_text(self, image_pil, text, box, font_color="black"):
        draw = ImageDraw.Draw(image_pil)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        font_size = self.get_font_size_that_fits(text, draw, w, h)
        font = ImageFont.truetype(self.font_path, font_size)
        
        # Wrap text again with the best font size
        avg_char_width = sum(draw.textlength(c, font=font) for c in "abcdefghijklmnopqrstuvwxyz")/26
        chars_per_line = max(1, int(w / avg_char_width))
        wrapped_text = textwrap.fill(text, width=chars_per_line)
        
        # Center the text in the box
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_x = x1 + (w - text_w) / 2
        text_y = y1 + (h - text_h) / 2
        
        draw.multiline_text(
            (text_x, text_y), 
            wrapped_text, 
            fill=font_color, 
            font=font, 
            align="center"
        )
        return image_pil

if __name__ == "__main__":
    # test
    img = Image.new('RGB', (200, 200), color='white')
    renderer = MangaRenderer()
    box = (10, 10, 190, 190)
    renderer.render_text(img, "Testing!!!!", box)
    img.save("test_render.png")
