import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas

from .. import config
from ..schemas import ModelSummary, ProblemType
from ..utils.state import JSONStore


@dataclass
class ReportRecord:
    report_id: str
    path: str
    created_at: str
    context: Dict


class ReportService:
    def __init__(self) -> None:
        config.ensure_directories()
        self._store = JSONStore(config.STORAGE_DIR / "reports_index.json")
    
    def _check_page_space(self, c, y_pos, margin, needed_space, page_number, add_footer):
        """Check if we have enough space on current page, add new page if not."""
        if y_pos - needed_space < margin + 20:
            add_footer(page_number)
            c.showPage()
            return page_number + 1, A4[1] - margin
        return page_number, y_pos
    
    def _add_footer(self, c, width, height, margin, page_num):
        """Add footer to page."""
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.gray)
        c.drawRightString(width - margin, margin / 2, f"Page {page_num}")
        c.setFillColor(colors.black)
        c.restoreState()
    
    def _draw_wrapped_text(self, c, x, y, text, max_width, font_size=10, line_spacing=4, align="left"):
        """Draw text with wrapping if it exceeds max_width."""
        c.saveState()
        c.setFont("Helvetica", font_size)
        c.setFillColor(colors.black)  # Ensure text is visible
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = c.stringWidth(word, "Helvetica", font_size)
            if current_width + word_width + c.stringWidth(" ", "Helvetica", font_size) <= max_width:
                current_line.append(word)
                current_width += word_width + c.stringWidth(" ", "Helvetica", font_size)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
        
        if current_line:
            lines.append(" ".join(current_line))
        
        for i, line in enumerate(lines):
            if align == "center":
                line_width = c.stringWidth(line, "Helvetica", font_size)
                x_pos = x + (max_width - line_width) / 2
            elif align == "right":
                line_width = c.stringWidth(line, "Helvetica", font_size)
                x_pos = x + max_width - line_width
            else:
                x_pos = x
            c.drawString(x_pos, y, line)
            y -= font_size + line_spacing
        
        c.restoreState()
        return y
    
    def _draw_centered_text(self, c, text, width, y, font_name="Helvetica", font_size=10, color=colors.black):
        """Draw centered text at specified y position."""
        c.saveState()
        c.setFont(font_name, font_size)
        c.setFillColor(color)
        text_width = c.stringWidth(text, font_name, font_size)
        x = (width - text_width) / 2
        c.drawString(x, y, text)
        c.restoreState()
        return y - (font_size + 4)
    
    def _draw_centered_metric_row(self, c, column_left_x, column_width, y, label, value, font_size=10):
        """Draw a metric row with consistent formatting, centered in its column."""
        c.saveState()
        c.setFont("Helvetica", font_size)
        c.setFillColor(colors.black)  # Ensure text is visible
        label_width = c.stringWidth(f"{label}:", "Helvetica", font_size)
        value_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        c.setFont("Helvetica-Bold", font_size)
        value_width = c.stringWidth(value_str, "Helvetica-Bold", font_size)
        
        total_width = label_width + value_width + 10
        start_x = column_left_x + (column_width - total_width) / 2
        
        c.setFont("Helvetica", font_size)
        c.drawString(start_x, y, f"{label}:")
        c.setFont("Helvetica-Bold", font_size)
        c.drawString(start_x + label_width + 10, y, value_str)
        c.restoreState()
        return y - (font_size + 4)
    
    def _add_compact_title_page(self, c, width, height, margin, page_number, 
                               leaderboard, target_column, problem_type, 
                               dataset_id=None, dataset_mode=None, intensity=None):
        """
        Create compact page 1 with BOTH:
        - Run summary + best model
        - Performance summary + top models table
        Everything the user asked for lives strictly on this first page.
        """
        accent_blue = colors.HexColor("#1D4ED8")
        accent_light = colors.HexColor("#EEF2FF")
        text_primary = colors.HexColor("#111827")
        text_muted = colors.HexColor("#6B7280")
        
        # Header - more compact
        header_h = 60
        c.setFillColor(accent_blue)
        c.rect(0, height - header_h, width, header_h, fill=1, stroke=0)
        
        # Centered title text
        self._draw_centered_text(c, "Model Evaluation Report", width, height - 30, 
                                "Helvetica-Bold", 18, colors.white)
        self._draw_centered_text(c, "AI-as-a-Service Platform", width, height - 45, 
                                "Helvetica", 9, colors.white)
        
        # Main content starts right after header
        y = height - header_h - 40  # More space after header
        
        # Run info in compact box
        c.setFillColor(accent_light)
        c.setStrokeColor(colors.whitesmoke)
        c.setLineWidth(0.5)
        box_height = 115 if intensity else 100  # Taller if intensity is shown
        box_width = width - 2 * margin
        c.roundRect(margin, y - box_height, box_width, box_height, 4, fill=1, stroke=1)
        
        # Centered section title
        self._draw_centered_text(c, "RUN SUMMARY", width, y - 15, 
                                "Helvetica-Bold", 10, accent_blue)
        
        # Compact info rows - centered
        info_y = y - 40
        rows = [
            f"Target: {target_column[:40]}{'...' if len(target_column) > 40 else ''}",
            f"Problem Type: {problem_type.title()}",
            f"Models Evaluated: {len(leaderboard)}",
        ]
        # Add training intensity if available
        if intensity:
            intensity_display = intensity.title()  # Capitalize first letter
            rows.append(f"Training Intensity: {intensity_display}")
        rows.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        
        c.setFillColor(text_primary)
        c.setFont("Helvetica", 9)
        for row in rows:
            self._draw_centered_text(c, row, width, info_y, "Helvetica", 9, text_primary)
            info_y -= 16  # More spacing between rows
        
        y = y - box_height - 40  # More space between sections
        
        # Best model highlight if available (skip for image datasets)
        if leaderboard and dataset_mode != "supervised" and dataset_mode != "unsupervised":
            best = leaderboard[0]
            metrics = best.get("metrics", {}) or {}
            
            c.setFillColor(colors.white)
            c.setStrokeColor(accent_blue)
            c.setLineWidth(0.8)
            highlight_h = 85  # Taller for better spacing
            highlight_width = width - 2 * margin
            c.roundRect(margin, y - highlight_h, highlight_width, highlight_h, 4, fill=1, stroke=1)
            
            # Centered section title
            self._draw_centered_text(c, "BEST MODEL", width, y - 20, 
                                    "Helvetica-Bold", 10, accent_blue)
            
            # Model name centered
            model_name = best.get("model_name", "Top Model")
            if len(model_name) > 50:
                model_name = model_name[:47] + "..."
            self._draw_centered_text(c, model_name, width, y - 40, 
                                    "Helvetica-Bold", 11, text_primary)
            
           
            
            # Compact metrics display - centered
            metrics_y = y - 55
            if problem_type == "classification":
                acc = metrics.get("accuracy")
                f1 = metrics.get("f1")
                if acc is not None:
                    self._draw_centered_text(c, f"Accuracy: {acc:.4f}", width, metrics_y, 
                                            "Helvetica", 9, text_primary)
                    metrics_y -= 15
                if f1 is not None:
                    self._draw_centered_text(c, f"F1 Score: {f1:.4f}", width, metrics_y, 
                                            "Helvetica", 9, text_primary)
            else:
                rmse = metrics.get("rmse")
                r2 = metrics.get("r2")
                if rmse is not None:
                    self._draw_centered_text(c, f"RMSE: {rmse:.4f}", width, metrics_y, 
                                            "Helvetica", 9, text_primary)
                    metrics_y -= 15
                if r2 is not None:
                    self._draw_centered_text(c, f"R²: {r2:.4f}", width, metrics_y, 
                                            "Helvetica", 9, text_primary)
            y = y - highlight_h - 50  # Move down after best model box
        
        # For image datasets, show sample images from each class
        if dataset_mode == "supervised" and dataset_id:
            try:
                from ..services.dataset_manager import DatasetManager
                dataset_mgr = DatasetManager()
                dataset_path, dataset_record = dataset_mgr.get_image_paths(dataset_id)
                
                # Get classes from dataset record
                classes = dataset_record.classes if hasattr(dataset_record, 'classes') else []
                
                if classes:
                    y -= 20  # Space before image section
                    self._draw_centered_text(c, "Sample Images by Class", width, y, 
                                            "Helvetica-Bold", 12, colors.black)
                    y -= 25
                    
                    # Display one image from each class
                    img_size = 80  # Size for each image
                    images_per_row = 3  # Show 3 images per row
                    spacing = 20
                    total_width = (img_size * images_per_row) + (spacing * (images_per_row - 1))
                    start_x = (width - total_width) / 2
                    
                    current_x = start_x
                    row_y = y
                    images_in_row = 0
                    last_img_y = row_y  # Track last image y position
                    
                    for class_info in classes[:9]:  # Limit to 9 classes max
                        class_label = class_info.get("label", "Unknown")
                        class_dir = dataset_path / class_label
                        
                        if class_dir.exists() and class_dir.is_dir():
                            # Find first image in class directory
                            from PIL import Image as PILImage
                            image_files = []
                            for ext in [".jpg", ".jpeg", ".png"]:
                                image_files.extend(list(class_dir.glob(f"*{ext}")))
                                image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
                            
                            if image_files:
                                img_path = image_files[0]
                                try:
                                    # Draw class label above image
                                    label_y = row_y
                                    c.setFont("Helvetica", 8)
                                    c.setFillColor(colors.black)
                                    label_width = c.stringWidth(class_label, "Helvetica", 8)
                                    label_x = current_x + (img_size - label_width) / 2
                                    c.drawString(label_x, label_y, class_label[:15])  # Truncate long labels
                                    
                                    # Draw image
                                    img_y = label_y - 15 - img_size
                                    last_img_y = img_y  # Update last image position
                                    try:
                                        c.drawImage(str(img_path), current_x, img_y, 
                                                   width=img_size, height=img_size, 
                                                   preserveAspectRatio=True, anchor='sw')
                                    except Exception:
                                        # If image fails to load, draw a placeholder box
                                        c.setStrokeColor(colors.gray)
                                        c.setFillColor(colors.lightgrey)
                                        c.rect(current_x, img_y, img_size, img_size, fill=1, stroke=1)
                                    
                                    # Move to next position
                                    current_x += img_size + spacing
                                    images_in_row += 1
                                    
                                    # Move to next row if needed
                                    if images_in_row >= images_per_row:
                                        current_x = start_x
                                        row_y = img_y - 30  # Space for next row
                                        images_in_row = 0
                                        
                                        # Check if we need a new page
                                        if row_y < margin + 100:
                                            self._add_footer(c, width, height, margin, page_number)
                                            c.showPage()
                                            page_number += 1
                                            row_y = height - margin - 50
                                except Exception:
                                    continue
                    
                    # Update y position after images
                    if images_in_row > 0:
                        y = last_img_y - 30
                    else:
                        y = row_y
                    y -= 20  # Extra space after images
            except Exception:
                # If image loading fails, just continue
                pass

        # -------------------------------------------------------------
        # Performance Summary (Executive Summary) ON THE SAME PAGE
        # -------------------------------------------------------------
        # Move cursor down
        if dataset_mode == "supervised" or dataset_mode == "unsupervised":
            y = y - 20  # Just a bit of space for image datasets
        elif leaderboard:
            y = y - 50  # Space after best model box for CSV
        else:
            y = y - 40

        # Centered section title
        self._draw_centered_text(c, "Performance Summary", width, y, 
                                "Helvetica-Bold", 14, colors.black)
        
        # Underline
        c.setStrokeColor(colors.darkblue)
        c.setLineWidth(1)
        line_width = 180
        line_x = (width - line_width) / 2
        c.line(line_x, y - 6, line_x + line_width, y - 6)
        y -= 30  # More space after title

        # Summary sentence - centered and wrapped
        total_models = len(leaderboard)
        if dataset_mode == "supervised" or dataset_mode == "unsupervised":
            summary_text = (
                f"This report summarizes the performance of the trained model for "
                f"{problem_type} tasks on image data."
            )
        else:
            summary_text = (
                f"This report summarizes the performance of {total_models} model"
                f"{'s' if total_models != 1 else ''} evaluated for "
                f"{problem_type} tasks."
            )
        y = self._draw_wrapped_text(c, margin, y, summary_text, width - 2 * margin, 
                                    11, 3, align="center")
        y -= 20  # Space before table

        # Skip table for image datasets (only one model)
        if dataset_mode != "supervised" and dataset_mode != "unsupervised":
            # Top models table header (rank/model/score/metric) – compact and centered
            # Define consistent table layout for alignment with Dataset Preview
            table_total_width = width - 2 * margin  # Full width minus margins
            
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(colors.black)  # Ensure text is visible
            # Calculate column positions for centered alignment within table
            rank_width = 40
            model_width = 160
            score_width = 60
            metric_width = 70
            table_width = rank_width + model_width + score_width + metric_width  # Total table width = 330
            
            # Center the table on the page
            table_margin = (width - table_width) / 2
            
            # Draw headers centered in their columns
            c.drawString(table_margin, y, "Rank")
            c.drawString(table_margin + rank_width + (model_width - c.stringWidth("Model", "Helvetica-Bold", 11)) / 2, y, "Model")
            c.drawString(table_margin + rank_width + model_width + (score_width - c.stringWidth("Score", "Helvetica-Bold", 11)) / 2, y, "Score")
            c.drawString(table_margin + rank_width + model_width + score_width + (metric_width - c.stringWidth("Key Metric", "Helvetica-Bold", 11)) / 2, y, "Key Metric")
            
            y -= 12  # More space before line
            c.setLineWidth(0.5)
            c.setStrokeColor(colors.darkblue)
            c.line(table_margin, y, table_margin + table_width, y)
            y -= 15  # More space after line before data rows

            # Single-model or top models table rows (still on page 1)
            c.setFont("Helvetica", 11)
            c.setFillColor(colors.black)  # Ensure text is visible
            for idx, model in enumerate(leaderboard[:5], 1):
                # Model name (trimmed to keep on line)
                model_name = model.get("model_name", "Unknown")
                if len(model_name) > 35:
                    model_name = model_name[:32] + "..."

                metrics = model.get("metrics", {})
                if problem_type == "classification":
                    key_metric = metrics.get("accuracy", metrics.get("f1", 0.0) or 0.0)
                    metric_name = "Accuracy" if "accuracy" in metrics else "F1"
                else:
                    key_metric = metrics.get("r2", metrics.get("rmse", 0.0) or 0.0)
                    metric_name = "R²" if "r2" in metrics else "RMSE"
                
                metric_value = f"{key_metric:.4f}" if isinstance(key_metric, (int, float)) else str(key_metric)

                # Draw centered in columns
                c.drawString(table_margin + (rank_width - c.stringWidth(str(idx), "Helvetica", 11)) / 2, y, str(idx))
                
                model_x = table_margin + rank_width + (model_width - c.stringWidth(model_name, "Helvetica", 11)) / 2
                c.drawString(model_x, y, model_name)
                
                score_x = table_margin + rank_width + model_width + (score_width - c.stringWidth(metric_value, "Helvetica", 11)) / 2
                c.drawString(score_x, y, metric_value)
                
                metric_x = table_margin + rank_width + model_width + score_width + (metric_width - c.stringWidth(metric_name, "Helvetica", 11)) / 2
                c.drawString(metric_x, y, metric_name)
                
                y -= 14  # More spacing between rows
        else:
            # For image datasets, show model metrics directly
            if leaderboard:
                best = leaderboard[0]
                metrics = best.get("metrics", {}) or {}
                model_name = best.get("model_name", "Model")
                
                c.setFont("Helvetica-Bold", 11)
                c.setFillColor(colors.black)
                self._draw_centered_text(c, f"Model: {model_name}", width, y, 
                                        "Helvetica-Bold", 11, colors.black)
                y -= 20
                
                c.setFont("Helvetica", 10)
                if problem_type == "classification":
                    acc = metrics.get("accuracy")
                    f1 = metrics.get("f1")
                    if acc is not None:
                        self._draw_centered_text(c, f"Accuracy: {acc:.4f}", width, y, 
                                                "Helvetica", 10, colors.black)
                        y -= 18
                    if f1 is not None:
                        self._draw_centered_text(c, f"F1 Score: {f1:.4f}", width, y, 
                                                "Helvetica", 10, colors.black)
                        y -= 18
                else:
                    rmse = metrics.get("rmse")
                    r2 = metrics.get("r2")
                    if rmse is not None:
                        self._draw_centered_text(c, f"RMSE: {rmse:.4f}", width, y, 
                                                "Helvetica", 10, colors.black)
                        y -= 18
                    if r2 is not None:
                        self._draw_centered_text(c, f"R²: {r2:.4f}", width, y, 
                                                "Helvetica", 10, colors.black)
                        y -= 18

        # Dataset Preview (only for CSV datasets)
        if dataset_mode == "csv" and dataset_id:
            y -= 25  # More space before dataset section
            self._draw_centered_text(c, "Dataset Preview (First 10 Rows)", width, y, 
                                    "Helvetica-Bold", 12, colors.black)
            y -= 20
            
            try:
                from ..services.dataset_manager import DatasetManager
                dataset_mgr = DatasetManager()
                df = dataset_mgr.load_dataframe(dataset_id)
                
                # Get first 10 rows
                preview_df = df.head(10)
                columns = preview_df.columns.tolist()
                
                # Limit number of columns to fit on page (max 6 columns)
                if len(columns) > 6:
                    columns = columns[:6]
                    preview_df = preview_df[columns]
                
                # Center the Dataset Preview table
                num_cols = len(columns)
                # Use same column widths as Performance Summary if same number of columns
                if num_cols == 4:
                    # Align with Performance Summary table - use same column widths
                    col_widths = [rank_width, model_width, score_width, metric_width]
                    dataset_table_margin = table_margin
                else:
                    # Center separately if different number of columns
                    dataset_table_width = table_total_width * 0.8  # Use 80% of available width
                    col_width = dataset_table_width / num_cols if num_cols > 0 else 0
                    col_widths = [col_width] * num_cols
                    dataset_table_margin = (width - dataset_table_width) / 2
                
                # Draw table header - centered in each column
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(colors.black)  # Ensure text is visible
                header_y = y
                x_pos = dataset_table_margin
                for idx, col in enumerate(columns):
                    col_width = col_widths[idx]
                    # Truncate column names if too long
                    col_name = col[:12] + "..." if len(col) > 12 else col
                    col_x = x_pos + (col_width - c.stringWidth(col_name, "Helvetica-Bold", 10)) / 2
                    c.drawString(col_x, header_y, col_name)
                    x_pos += col_width
                
                y -= 12
                c.setLineWidth(0.3)
                dataset_table_width = sum(col_widths)
                c.line(dataset_table_margin, y, dataset_table_margin + dataset_table_width, y)
                y -= 15  # More space below the separator line before first data row
                
                # Draw data rows - centered in each column
                c.setFont("Helvetica", 9)
                c.setFillColor(colors.black)  # Ensure text is visible
                for row_idx in range(len(preview_df)):
                    if y < margin + 40:  # Check if we need more space
                        break
                    row = preview_df.iloc[row_idx]
                    x_pos = dataset_table_margin
                    for col_idx, col in enumerate(columns):
                        col_width = col_widths[col_idx]
                        cell_value = str(row[col])
                        # Truncate cell values if too long
                        if len(cell_value) > 15:
                            cell_value = cell_value[:12] + "..."
                        cell_x = x_pos + (col_width - c.stringWidth(cell_value, "Helvetica", 9)) / 2
                        c.drawString(cell_x, y, cell_value)
                        x_pos += col_width
                    y -= 12
            except Exception:
                # If dataset loading fails, just skip the preview
                pass

        # Add footer (Page 1)
        self._add_footer(c, width, height, margin, page_number)
        c.showPage()
        return page_number + 1
    
    def _add_executive_summary(self, c, width, height, margin, page_number, 
                              leaderboard, problem_type):
        """Add executive summary with efficient space usage."""
        y = height - margin
        
        self._draw_centered_text(c, "Performance Summary", width, y, 
                                "Helvetica-Bold", 16, colors.black)
        # Centered underline
        c.setStrokeColor(colors.darkblue)
        c.setLineWidth(1)
        line_width = 200
        line_x = (width - line_width) / 2
        c.line(line_x, y - 8, line_x + line_width, y - 8)
        
        y -= 35  # More space after title
        
        # Summary text - centered
        total_models = len(leaderboard)
        summary_text = f"This report summarizes the performance of {total_models} model{'s' if total_models > 1 else ''} "
        summary_text += f"evaluated for {problem_type} tasks."
        
        y = self._draw_wrapped_text(c, margin, y, summary_text, width - 2 * margin, 10, 4, align="center")
        y -= 25  # More space before next section
        
        # Top models table title - centered
        self._draw_centered_text(c, "Top Performing Models", width, y, 
                                "Helvetica-Bold", 12, colors.darkblue)
        y -= 25  # More space after title
        
        # Draw table headers - centered in their columns
        c.setFont("Helvetica-Bold", 10)
        table_start_x = (width - 300) / 2  # Center the table
        
        c.drawString(table_start_x, y, "Rank")
        c.drawString(table_start_x + 40, y, "Model")
        c.drawString(table_start_x + 200, y, "Score")
        c.drawString(table_start_x + 270, y, "Key Metric")
        
        y -= 16  # More space before line
        c.setLineWidth(0.5)
        c.setStrokeColor(colors.darkblue)
        c.line(table_start_x, y, table_start_x + 300, y)
        y -= 15  # More space after line before data rows
        
        # Table rows
        c.setFont("Helvetica", 9)
        for idx, model in enumerate(leaderboard[:10], 1):
            # Check space for this row
            if y < margin + 40:
                self._add_footer(c, width, height, margin, page_number)
                c.showPage()
                page_number += 1
                y = height - margin - 30
                # Redraw headers on new page
                table_start_x = (width - 300) / 2
                c.setFont("Helvetica-Bold", 10)
                c.drawString(table_start_x, y, "Rank")
                c.drawString(table_start_x + 40, y, "Model")
                c.drawString(table_start_x + 200, y, "Score")
                c.drawString(table_start_x + 270, y, "Key Metric")
                y -= 16
                c.setLineWidth(0.5)
                c.setStrokeColor(colors.darkblue)
                c.line(table_start_x, y, table_start_x + 300, y)
                y -= 15
                c.setFont("Helvetica", 9)
            
            # Model name (truncated if needed)
            model_name = model.get("model_name", "Unknown")
            if len(model_name) > 40:
                model_name = model_name[:37] + "..."
            
            # Get key metric
            metrics = model.get("metrics", {})
            if problem_type == "classification":
                key_metric = metrics.get("accuracy", metrics.get("f1", 0))
                metric_name = "Accuracy" if "accuracy" in metrics else "F1"
            else:
                key_metric = metrics.get("r2", metrics.get("rmse", 0))
                metric_name = "R²" if "r2" in metrics else "RMSE"
            
            metric_value = f"{key_metric:.4f}" if isinstance(key_metric, (int, float)) else str(key_metric)
            
            # Center text in columns
            rank_x = table_start_x + (40 - c.stringWidth(str(idx), "Helvetica", 9)) / 2
            c.drawString(rank_x, y, str(idx))
            
            model_x = table_start_x + 40 + (160 - c.stringWidth(model_name, "Helvetica", 9)) / 2
            c.drawString(model_x, y, model_name)
            
            score_x = table_start_x + 200 + (70 - c.stringWidth(metric_value, "Helvetica", 9)) / 2
            c.drawString(score_x, y, metric_value)
            
            metric_x = table_start_x + 270 + (30 - c.stringWidth(metric_name, "Helvetica", 9)) / 2
            c.drawString(metric_x, y, metric_name)
            
            y -= 16  # More spacing between rows
        
        y -= 25  # More space after table
        
        # Add footer
        self._add_footer(c, width, height, margin, page_number)
        c.showPage()
        return page_number + 1
    
    def _add_detailed_model_page(self, c, width, height, margin, page_number, 
                                 model_info, problem_type, model_index, total_models):
        """Add detailed model information with dynamic layout and centered alignment."""
        y = height - margin - 20  # Start lower for better top margin
        
        # Model header with rank - centered
        rank = model_info.get('rank', model_index + 1)
        model_name = model_info.get("model_name", f"Model {rank}")
        if len(model_name) > 50:
            model_name = model_name[:47] + "..."
        
        self._draw_centered_text(c, f"{rank}. {model_name}", width, y, 
                                "Helvetica-Bold", 16, colors.black)
        
        # Model key - centered
        model_key = model_info.get('model_key', 'N/A')
        if len(model_key) > 60:
            model_key = model_key[:57] + "..."
        y = self._draw_centered_text(c, f"Key: {model_key}", width, y - 20, 
                                    "Helvetica", 10, colors.black)
        
        y -= 30  # More space before separator
        
        # Horizontal separator - centered
        c.setStrokeColor(colors.gray)
        c.setLineWidth(0.5)
        separator_width = width - 2 * margin
        separator_x = (width - separator_width) / 2
        c.line(separator_x, y, separator_x + separator_width, y)
        
        y -= 35  # More space after separator
        
        # Two-column layout: Performance Metrics (left) and Feature Overview (right)
        content_width = width - 2 * margin
        column_width = (content_width - 40) / 2  # 40px gap between columns for better spacing
        left_x = margin
        right_x = margin + column_width + 40
        
        # Store starting y for both columns
        section_start_y = y
        
        # Left column: Performance Metrics - centered in column
        metrics_title_x = left_x + (column_width - c.stringWidth("Performance Metrics", "Helvetica-Bold", 12)) / 2
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.darkblue)
        c.drawString(metrics_title_x, y, "Performance Metrics")
        c.setFillColor(colors.black)
        
        y -= 25  # Space after title
        
        metrics = model_info.get("metrics", {})
        if problem_type == "classification":
            metric_items = [
                ("Accuracy", metrics.get("accuracy")),
                ("Precision", metrics.get("precision")),
                ("Recall", metrics.get("recall")),
                ("F1 Score", metrics.get("f1")),
            ]
        else:
            metric_items = [
                ("RMSE", metrics.get("rmse")),
                ("R² Score", metrics.get("r2")),
                ("MAE", metrics.get("mae")),
            ]
        
        left_y = y
        for label, value in metric_items:
            if value is not None:
                left_y = self._draw_centered_metric_row(c, left_x, column_width, left_y, label, value, font_size=10)
                left_y -= 8  # Space between metric rows
        
        # Right column: Feature Overview - centered in column
        feature_hints = model_info.get("feature_hints", [])
        right_y = section_start_y - 25  # Align with metrics title
        
        if feature_hints:
            feature_title_x = right_x + (column_width - c.stringWidth("Feature Overview", "Helvetica-Bold", 12)) / 2
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(colors.darkblue)
            c.drawString(feature_title_x, right_y, "Feature Overview")
            c.setFillColor(colors.black)
            right_y -= 25  # Space after title
            
            c.setFont("Helvetica", 9)
            for hint in feature_hints[:8]:  # Limit to 8 most important
                hint_name = hint.get("name", "Unknown")
                hint_kind = hint.get("kind", "unknown")
                
                if hint_kind == "categorical":
                    value_map = hint.get("value_map", {})
                    if value_map:
                        mappings = ", ".join(f"{k}={v}" for k, v in list(value_map.items())[:3])
                        if len(value_map) > 3:
                            mappings += f"... (+{len(value_map)-3})"
                        text = f"• {hint_name}: {mappings}"
                elif hint_kind == "numeric":
                    text = f"• {hint_name}: numeric"
                else:
                    text = f"• {hint_name}: {hint_kind}"
                
                # Draw feature hint centered in column
                text_width = c.stringWidth(text, "Helvetica", 9)
                hint_x = right_x + (column_width - text_width) / 2
                c.drawString(hint_x, right_y, text)
                right_y -= 15  # Space between feature hints
        
        # Move y to the lower of the two columns
        y = min(left_y, right_y) - 35  # More space before next section
        
        # Visualizations section - centered
        plots = model_info.get("plots", {})
        if plots:
            self._draw_centered_text(c, "Visualizations", width, y, 
                                    "Helvetica-Bold", 12, colors.darkblue)
            y -= 25  # Space after title
            
            # Check available space
            available_height = y - margin - 60  # More margin at bottom
            if available_height < 180:
                self._add_footer(c, width, height, margin, page_number)
                c.showPage()
                page_number += 1
                y = height - margin - 30
                self._draw_centered_text(c, "Visualizations", width, y, 
                                        "Helvetica-Bold", 12, colors.darkblue)
                y -= 25
                available_height = y - margin - 60
            
            # Two-column layout for images
            content_width = width - 2 * margin
            column_width = (content_width - 40) / 2  # 40px gap for better spacing
            left_x = margin
            right_x = margin + column_width + 40
            
            # Calculate image height
            img_height = min(200, available_height * 0.5)  # Smaller percentage for better fit
            
            # Track if any images were drawn
            images_drawn = False
            
            # Left: Confusion Matrix
            confusion_path = plots.get("confusion_plot")
            if confusion_path:
                confusion_img = Path(confusion_path)
                if confusion_img.exists():
                    images_drawn = True
                    # Label centered in column
                    label_x = left_x + (column_width - c.stringWidth("Confusion Matrix", "Helvetica", 9)) / 2
                    c.setFont("Helvetica", 9)
                    c.drawString(label_x, y, "Confusion Matrix")
                    
                    # Image centered in column
                    try:
                        c.drawImage(
                            str(confusion_img),
                            left_x + (column_width - column_width * 0.9) / 2,
                            y - 15 - img_height,
                            width=column_width * 0.9,
                            height=img_height,
                            preserveAspectRatio=True,
                            anchor='c'
                        )
                    except Exception:
                        pass
            
            # Right: Feature Importance
            feature_imp_path = plots.get("feature_importance_plot")
            if feature_imp_path:
                feature_imp_img = Path(feature_imp_path)
                if feature_imp_img.exists():
                    images_drawn = True
                    # Label centered in column
                    label_x = right_x + (column_width - c.stringWidth("Feature Importance", "Helvetica", 9)) / 2
                    c.setFont("Helvetica", 9)
                    c.drawString(label_x, y, "Feature Importance")
                    
                    # Image centered in column
                    try:
                        c.drawImage(
                            str(feature_imp_img),
                            right_x + (column_width - column_width * 0.9) / 2,
                            y - 15 - img_height,
                            width=column_width * 0.9,
                            height=img_height,
                            preserveAspectRatio=True,
                            anchor='c'
                        )
                    except Exception:
                        pass
            
            # Move y down below the images (only if at least one image was drawn)
            if images_drawn:
                y = y - 15 - img_height - 35  # More space after images
            else:
                y -= 20  # Just a small space if no images
            
            # Other visualizations (ROC, Residuals, Tree, SHAP) below in full width
            other_plots = [
                ("roc_plot", "ROC Curve"),
                ("residual_plot", "Residuals"),
                ("tree_plot", "Decision Tree Structure"),
                ("shap_plot", "SHAP Summary"),
            ]
            
            for plot_key, caption_label in other_plots:
                plot_path = plots.get(plot_key)
                if not plot_path:
                    continue
                    
                img_path = Path(plot_path)
                if not img_path.exists():
                    continue
                
                available_height = y - margin - 60
                # Decision Tree needs more space, so check for higher minimum
                min_height_required = 200 if plot_key == "tree_plot" else 120
                if available_height < min_height_required:
                    self._add_footer(c, width, height, margin, page_number)
                    c.showPage()
                    page_number += 1
                    y = height - margin - 30
                    y -= 30
                    available_height = y - margin - 60
                
                if plot_key == "tree_plot":
                    # Decision Tree Structure - make it bigger
                    img_height = min(400, available_height * 0.85)
                elif plot_key in {"confusion_plot", "shap_plot"}:
                    img_height = min(220, available_height * 0.7)
                elif plot_key in {"roc_plot", "residual_plot"}:
                    img_height = min(200, available_height * 0.6)
                else:
                    img_height = min(180, available_height * 0.5)
                
                # Centered caption (skip for Decision Tree Structure)
                if plot_key != "tree_plot":
                    self._draw_centered_text(c, caption_label, width, y, 
                                            "Helvetica", 9, colors.black)
                    y -= 15
                
                # Centered image
                try:
                    img_width = width - 2 * margin
                    # Use more width for Decision Tree Structure
                    width_factor = 0.95 if plot_key == "tree_plot" else 0.9
                    c.drawImage(
                        str(img_path),
                        margin + (img_width - img_width * width_factor) / 2,
                        y - img_height,
                        width=img_width * width_factor,
                        height=img_height,
                        preserveAspectRatio=True,
                        anchor='c'
                    )
                    y -= img_height + 30  # More space after image
                except Exception as e:
                    error_text = f"Failed to load image: {str(e)[:50]}"
                    self._draw_centered_text(c, error_text, width, y, 
                                            "Helvetica", 9, colors.red)
                    y -= 40
        
        # Add footer
        self._add_footer(c, width, height, margin, page_number)
        c.showPage()
        return page_number + 1
    
    def _add_code_usage_section(self, c, width, height, margin, page_number, 
                                best_model, problem_type, target_column, 
                                model_id=None, feature_columns=None, dataset_mode=None):
        """Add code usage instructions for the best model at the end of the report."""
        y = height - margin - 20
        
        # Title
        self._draw_centered_text(c, "How to Use Your Trained Model", width, y, 
                                "Helvetica-Bold", 16, colors.black)
        y -= 30
        
        # Model file paths (define early so they can be used in instructions)
        model_file = f"model_{model_id}.pkl" if model_id else "model.pkl"
        h5_file = f"model_{model_id}.h5" if model_id else "model.h5"
        
        # Instructions
        if model_id:
            instructions = [
                f"1. Download the model files (.pkl and/or .h5) from Training History.",
                f"2. Save the files as '{model_file}' and/or '{h5_file}' in your project directory.",
                "3. Install required packages: pip install pandas scikit-learn (and tensorflow for .h5 files).",
                "4. Copy and run the code below - it's ready to use with your specific model!",
            ]
        else:
            instructions = [
                "1. Download the model files (.pkl and/or .h5) from the Training History section.",
                "2. Save the files in your project directory and update the file names in the code.",
                "3. Install required packages: pip install pandas scikit-learn (and tensorflow for .h5 files).",
                "4. Use the code examples below to load and make predictions with your model.",
            ]
        
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        for instruction in instructions:
            y = self._draw_wrapped_text(c, margin, y, instruction, width - 2 * margin, 
                                       10, 3, align="center")
            y -= 8
        
        y -= 20
        
        # Model Information
        model_name = best_model.get("model_name", "Best Model")
        model_key = best_model.get("model_key", "unknown")
        feature_hints = best_model.get("feature_hints", [])
        
        # Extract feature names (excluding target column)
        if feature_columns:
            feature_names = [col for col in feature_columns if col != target_column]
        elif feature_hints:
            feature_names = [hint.get("name", "") for hint in feature_hints if hint.get("name") != target_column]
        else:
            feature_names = []
        
        c.setFont("Helvetica-Bold", 12)
        self._draw_centered_text(c, f"Model: {model_name}", width, y, 
                                "Helvetica-Bold", 12, colors.darkblue)
        y -= 25
        
        # Code section for .pkl file (scikit-learn models)
        c.setFont("Helvetica-Bold", 11)
        self._draw_centered_text(c, "Using .pkl File (Recommended for most models)", width, y, 
                                "Helvetica-Bold", 11, colors.black)
        y -= 20
        
        # Generate code with input prompts for actual feature names
        if feature_names and dataset_mode == "csv":
            # Build input collection code for actual feature names
            input_lines = []
            for i, feat_name in enumerate(feature_names[:25]):  # Limit to 25 features for readability
                # Get feature type from hints if available
                feat_hint = next((h for h in feature_hints if h.get("name") == feat_name), {})
                feat_kind = feat_hint.get("kind", "numeric")
                
                if feat_kind == "numeric":
                    example_val = "0.0"  # Default numeric
                    if "example" in feat_hint:
                        example_val = str(feat_hint["example"])
                    input_lines.append(f"while True:")
                    input_lines.append(f"    try:")
                    input_lines.append(f"        {feat_name} = float(input(f'Enter {feat_name} (numeric, e.g. {example_val}): '))")
                    input_lines.append(f"        break")
                    input_lines.append(f"    except ValueError:")
                    input_lines.append(f"        print('Invalid input. Please enter a number.')")
                elif feat_kind == "categorical":
                    value_map = feat_hint.get("value_map", {})
                    if value_map:
                        valid_values = ", ".join(list(value_map.keys())[:5])
                        if len(value_map) > 5:
                            valid_values += f", ... (total {len(value_map)} categories)"
                        input_lines.append(f"{feat_name} = input(f'Enter {feat_name} (categorical, e.g. {valid_values}): ')")
                    else:
                        input_lines.append(f"{feat_name} = input(f'Enter {feat_name} (categorical): ')")
                else:
                    input_lines.append(f"{feat_name} = input(f'Enter {feat_name}: ')")
            
            if len(feature_names) > 25:
                input_lines.append(f"# ... and {len(feature_names) - 25} more features (add input prompts for them)")
            
            input_code = "\n".join(input_lines)
            
            # Build DataFrame creation code
            feature_dict_lines = ["# Create DataFrame from user inputs", "data = pd.DataFrame({"]
            for feat_name in feature_names[:25]:
                feature_dict_lines.append(f"    '{feat_name}': [{feat_name}],")
            if len(feature_names) > 25:
                feature_dict_lines.append(f"    # ... add remaining {len(feature_names) - 25} features here")
            feature_dict_lines.append("})")
            feature_dict = "\n".join(feature_dict_lines)
        else:
            input_code = "# Add input prompts for your features\nfeature1 = input('Enter feature1: ')\nfeature2 = input('Enter feature2: ')"
            feature_dict = "# Create DataFrame from user inputs\ndata = pd.DataFrame({\n    'feature1': [feature1],\n    'feature2': [feature2],\n})"
        
        # Code example for .pkl (adapt based on problem type and dataset mode)
        # For image datasets, .pkl contains a TensorFlow/Keras model wrapper
        if dataset_mode == "supervised" or dataset_mode == "unsupervised":
            # For image models
            pkl_code = [
                "import pickle",
                "import tensorflow as tf",
                "import numpy as np",
                "from PIL import Image",
                "",
                f"# Load the model (replace '{model_file}' with your downloaded file path)",
                f"with open('{model_file}', 'rb') as f:",
                "    model_package = pickle.load(f)",
                "",
                "# Extract the model from the package",
                "model = model_package.estimator",
                "",
                "# Get image path from user",
                "image_path = input('Enter the path to your image file: ')",
                "",
                "# Load and preprocess the image",
                "try:",
                "    image = Image.open(image_path)",
                "    image = image.convert('RGB')  # Ensure RGB format",
                "    image = image.resize((128, 128))  # Standard size used during training",
                "    image_array = np.array(image, dtype=np.float32) / 255.0",
                "    image_array = np.expand_dims(image_array, axis=0)",
                "    ",
                "    # Make prediction",
                "    prediction = model.predict(image_array, verbose=0)",
                "    predicted_class = np.argmax(prediction[0])",
                "    confidence = np.max(prediction[0])",
                "    print(f'\\nPredicted class: {predicted_class}')",
                "    print(f'Confidence: {confidence:.4f}')",
                "except FileNotFoundError:",
                "    print(f'Error: Image file not found at {image_path}')",
                "except Exception as e:",
                "    print(f'Error processing image: {e}')",
            ]
        elif problem_type == "classification":
            # For CSV classification models
            pkl_code = [
                "import pickle",
                "import pandas as pd",
                "",
                f"# Load the model (replace '{model_file}' with your downloaded file path)",
                f"with open('{model_file}', 'rb') as f:",
                "    model_package = pickle.load(f)",
                "",
                "# Extract the estimator from the package",
                "model = model_package.estimator",
                "pipeline = model_package.pipeline",
                "",
                "print('\\n=== Enter your data for prediction ===')",
                "# Get input from user for each feature",
            ] + input_code.split("\n") + [
                "",
            ] + feature_dict.split("\n") + [
                "",
                "",
                "# Ensure all required columns are present (model will handle missing ones)",
                "for col in model_package.columns:",
                "    if col not in data.columns:",
                "        data[col] = None",
                "",
                "# Reorder columns to match training order",
                "data = data[[col for col in model_package.columns if col in data.columns]]",
                "",
                "# Apply preprocessing pipeline and make predictions",
                "X_processed = pipeline.transform(data)",
                "predictions = model.predict(X_processed)",
                "",
                "# Decode predictions if label encoder was used",
                "if model_package.label_encoder is not None:",
                "    predictions = model_package.label_encoder.inverse_transform(predictions.astype(int))",
                "",
                "print(f'Predicted class: {predictions[0]}')",
                "",
                "# For probability scores (if available):",
                "try:",
                "    probabilities = model.predict_proba(X_processed)",
                "    print(f'Probabilities: {probabilities[0]}')",
                "except AttributeError:",
                "    print('Probability scores not available for this model')",
            ]
        else:  # regression
            pkl_code = [
                "import pickle",
                "import pandas as pd",
                "",
                f"# Load the model (replace '{model_file}' with your downloaded file path)",
                f"with open('{model_file}', 'rb') as f:",
                "    model_package = pickle.load(f)",
                "",
                "# Extract the estimator from the package",
                "model = model_package.estimator",
                "pipeline = model_package.pipeline",
                "",
                "print('\\n=== Enter your data for prediction ===')",
                "# Get input from user for each feature",
            ] + input_code.split("\n") + [
                "",
            ] + feature_dict.split("\n") + [
                "",
                "",
                "# Ensure all required columns are present (model will handle missing ones)",
                "for col in model_package.columns:",
                "    if col not in data.columns:",
                "        data[col] = None",
                "",
                "# Reorder columns to match training order",
                "data = data[[col for col in model_package.columns if col in data.columns]]",
                "",
                "# Apply preprocessing pipeline and make predictions",
                "X_processed = pipeline.transform(data)",
                "predictions = model.predict(X_processed)",
                "",
                f"print(f'Predicted {target_column}: {{predictions[0]:.4f}}')",
            ]
        
        # Draw code with monospace-like font (left-aligned to preserve indentation)
        code_start_y = y
        c.setFont("Courier", 7)  # Slightly smaller font for better fit
        c.setFillColor(colors.black)
        line_height = 9
        max_line_width = width - 2 * margin - 20  # Leave some margin
        code_x_pos = margin + 10  # Left-aligned position for code
        
        for line in pkl_code:
            if y < margin + 60:
                self._add_footer(c, width, height, margin, page_number)
                c.showPage()
                page_number += 1
                y = height - margin - 20
                c.setFont("Courier", 7)
            
            # Wrap long lines
            line_width = c.stringWidth(line, "Courier", 7)
            if line_width > max_line_width:
                # Split long lines (preserve indentation on continuation)
                words = line.split()
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    test_width = c.stringWidth(test_line, "Courier", 7)
                    if test_width > max_line_width and current_line:
                        # Draw current line
                        c.drawString(code_x_pos, y, current_line)
                        y -= line_height
                        # Preserve indentation for continuation line
                        indent = len(line) - len(line.lstrip())
                        current_line = " " * indent + word
                    else:
                        current_line = test_line
                if current_line:
                    c.drawString(code_x_pos, y, current_line)
                    y -= line_height
            else:
                # Left-aligned (preserves Python indentation)
                c.drawString(code_x_pos, y, line)
                y -= line_height
        
        y -= 20
        
        # Code section for .h5 file (TensorFlow/Keras models) - Always show both
        c.setFont("Helvetica-Bold", 11)
        self._draw_centered_text(c, "Using .h5 File (Alternative method)", width, y, 
                                "Helvetica-Bold", 11, colors.black)
        y -= 20
        
        # Generate h5 code based on dataset mode
        if dataset_mode == "csv":
            # For CSV models, .h5 contains the same model but in HDF5 format
            h5_code = [
                "import pickle",
                "import h5py",
                "import numpy as np",
                "import pandas as pd",
                "",
                f"# Load the model from .h5 file (replace '{h5_file}' with your downloaded file path)",
                f"with h5py.File('{h5_file}', 'r') as f:",
                "    pickle_data = f['model_pickle'][:]",
                "    model_package = pickle.loads(pickle_data.tobytes())",
                "",
                "# Extract the estimator from the package",
                "model = model_package.estimator",
                "pipeline = model_package.pipeline",
                "",
                "print('\\n=== Enter your data for prediction ===')",
                "# Get input from user for each feature",
            ] + input_code.split("\n") + [
                "",
            ] + feature_dict.split("\n") + [
                "",
                "# Ensure all required columns are present",
                "for col in model_package.columns:",
                "    if col not in data.columns:",
                "        data[col] = None",
                "",
                "# Reorder columns to match training order",
                "data = data[[col for col in model_package.columns if col in data.columns]]",
                "",
                "# Apply preprocessing pipeline and make predictions",
                "X_processed = pipeline.transform(data)",
                "predictions = model.predict(X_processed)",
                "",
            ]
            if problem_type == "classification":
                h5_code.extend([
                    "# Decode predictions if label encoder was used",
                    "if model_package.label_encoder is not None:",
                    "    predictions = model_package.label_encoder.inverse_transform(predictions.astype(int))",
                    "",
                    "print(f'Predicted class: {predictions[0]}')",
                    "",
                    "# For probability scores (if available):",
                    "try:",
                    "    probabilities = model.predict_proba(X_processed)",
                    "    print(f'Probabilities: {probabilities[0]}')",
                    "except AttributeError:",
                    "    print('Probability scores not available for this model')",
                ])
            else:
                h5_code.extend([
                    f"print(f'Predicted {target_column}: {{predictions[0]:.4f}}')",
                ])
        else:
            # For image models
            h5_code = [
                "import tensorflow as tf",
                "import numpy as np",
                "from PIL import Image",
                "",
                f"# Load the model (replace '{h5_file}' with your downloaded file path)",
                f"model = tf.keras.models.load_model('{h5_file}')",
                "",
                "# Get image path from user",
                "image_path = input('Enter the path to your image file: ')",
                "",
                "# Load and preprocess the image",
                "try:",
                "    image = Image.open(image_path)",
                "    image = image.convert('RGB')  # Ensure RGB format",
                "    image = image.resize((128, 128))  # Standard size used during training",
                "    image_array = np.array(image, dtype=np.float32) / 255.0",
                "    image_array = np.expand_dims(image_array, axis=0)",
                "    ",
                "    # Make prediction",
                "    prediction = model.predict(image_array, verbose=0)",
                "    predicted_class = np.argmax(prediction[0])",
                "    confidence = np.max(prediction[0])",
                "    print(f'\\nPredicted class: {predicted_class}')",
                "    print(f'Confidence: {confidence:.4f}')",
                "except FileNotFoundError:",
                "    print(f'Error: Image file not found at {image_path}')",
                "except Exception as e:",
                "    print(f'Error processing image: {e}')",
            ]
        
        c.setFont("Courier", 7)
        for line in h5_code:
            if y < margin + 60:
                self._add_footer(c, width, height, margin, page_number)
                c.showPage()
                page_number += 1
                y = height - margin - 20
                c.setFont("Courier", 7)
            
            # Wrap long lines (preserve indentation)
            line_width = c.stringWidth(line, "Courier", 7)
            if line_width > max_line_width:
                # Split long lines (preserve indentation on continuation)
                words = line.split()
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    test_width = c.stringWidth(test_line, "Courier", 7)
                    if test_width > max_line_width and current_line:
                        # Draw current line
                        c.drawString(code_x_pos, y, current_line)
                        y -= line_height
                        # Preserve indentation for continuation line
                        indent = len(line) - len(line.lstrip())
                        current_line = " " * indent + word
                    else:
                        current_line = test_line
                if current_line:
                    c.drawString(code_x_pos, y, current_line)
                    y -= line_height
            else:
                # Left-aligned (preserves Python indentation)
                c.drawString(code_x_pos, y, line)
                y -= line_height
        
        y -= 20
        
        # Additional notes
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.darkblue)
        if dataset_mode == "supervised" or dataset_mode == "unsupervised":
            # Image dataset notes
            if problem_type == "classification":
                notes = [
                    "Important Notes:",
                    "• Provide the path to a single image file (JPG, PNG, or JPEG format).",
                    "• The image will be automatically resized to 128x128 pixels and converted to RGB.",
                    "• Predictions will be class labels (numeric class indices).",
                    "• Confidence scores represent the model's certainty for the predicted class.",
                    "• Ensure your image is clear and similar to the training data for best results.",
                ]
            else:  # regression (unlikely for images, but handle it)
                notes = [
                    "Important Notes:",
                    "• Provide the path to a single image file (JPG, PNG, or JPEG format).",
                    "• The image will be automatically resized to 128x128 pixels and converted to RGB.",
                    "• Predictions will be numerical values (continuous numbers).",
                    "• Ensure your image is clear and similar to the training data for best results.",
                ]
        elif problem_type == "classification":
            notes = [
                "Important Notes:",
                "• Ensure your input data has the same features/columns as the training data.",
                "• Predictions will be class labels (same format as your target column).",
                "• Use predict_proba() to get probability scores for each class.",
                "• The model expects the same data preprocessing as during training.",
            ]
        else:  # regression
            notes = [
                "Important Notes:",
                "• Ensure your input data has the same features/columns as the training data.",
                "• Predictions will be numerical values (continuous numbers).",
                "• The model expects the same data preprocessing as during training.",
                f"• Target column used during training: {target_column}",
            ]
        
        for note in notes:
            if y < margin + 40:
                self._add_footer(c, width, height, margin, page_number)
                c.showPage()
                page_number += 1
                y = height - margin - 20
                c.setFont("Helvetica", 9)
            
            y = self._draw_wrapped_text(c, margin, y, note, width - 2 * margin, 
                                       9, 3, align="center")
            y -= 10
        
        # Add footer
        self._add_footer(c, width, height, margin, page_number)
        c.showPage()
        return page_number + 1

    def generate_top_report(
        self,
        leaderboard: List[Dict],
        target_column: str,
        problem_type: ProblemType,
        user_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_mode: Optional[str] = None,
        intensity: Optional[str] = None,
    ) -> ReportRecord:
        """Generate a PDF report with optimized layout and spacing."""
        report_id = str(uuid.uuid4())
        path = config.REPORTS_DIR / f"{report_id}.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        margin = 40
        
        page_number = 1
        
        # 1. Title page
        page_number = self._add_compact_title_page(
            c, width, height, margin, page_number,
            leaderboard, target_column, problem_type,
            dataset_id, dataset_mode, intensity
        )
        
        # 2. Detailed model pages (top 5 or fewer) start from page 2
        # Skip for image datasets (only one model, no need for detailed pages)
        if dataset_mode != "supervised" and dataset_mode != "unsupervised":
            max_models = min(5, len(leaderboard))
            for idx, model_info in enumerate(leaderboard[:max_models]):
                page_number = self._add_detailed_model_page(
                    c, width, height, margin, page_number,
                    model_info, problem_type, idx, max_models
                )
        
        # 3. Code usage section at the end (for best model only)
        if leaderboard:
            best_model = leaderboard[0]
            # Get model_id and feature information from context if available
            model_id = best_model.get("model_id")
            feature_columns = best_model.get("columns")
            if not feature_columns and best_model.get("feature_hints"):
                feature_columns = [hint.get("name") for hint in best_model.get("feature_hints", [])]
            
            page_number = self._add_code_usage_section(
                c, width, height, margin, page_number,
                best_model, problem_type, target_column,
                model_id=model_id, feature_columns=feature_columns, dataset_mode=dataset_mode
            )
        
        c.save()
        
        # Verify file creation
        if not path.exists():
            raise RuntimeError(f"Failed to create report file at {path}")
        
        # Store record
        absolute_path = path.resolve()
        record = ReportRecord(
            report_id=report_id,
            path=str(absolute_path),
            created_at=datetime.utcnow().isoformat(),
            context={
                "leaderboard": leaderboard,
                "target_column": target_column,
                "problem_type": problem_type,
                "user_id": user_id,
                "dataset_id": dataset_id,
                "dataset_mode": dataset_mode,
            },
        )
        self._store.set(report_id, record.__dict__)
        return record

    def get(self, report_id: str) -> ReportRecord:
        """Retrieve a report record by ID."""
        payload = self._store.get(report_id)
        if not payload:
            all_keys = list(self._store.as_dict().keys())
            raise ValueError(
                f"Report {report_id} not found. "
                f"Available: {len(all_keys)} reports. "
                f"First few IDs: {all_keys[:3] if all_keys else 'none'}"
            )
        
        try:
            record = ReportRecord(**payload)
            report_path = Path(record.path)
            
            # Ensure absolute path
            if not report_path.is_absolute():
                report_path = config.REPORTS_DIR / report_path.name
            
            # Check existence
            if not report_path.exists():
                report_path = config.REPORTS_DIR / f"{report_id}.pdf"
                if not report_path.exists():
                    raise ValueError(f"Report file not found: {report_id}")
            
            record.path = str(report_path.resolve())
            return record
        except Exception as exc:
            raise ValueError(f"Failed to load report {report_id}: {str(exc)}")