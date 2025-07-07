import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import itertools
import matplotx
from style.genevive import genevive
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib import font_manager

# Font load
font_files = font_manager.findSystemFonts(
    fontpaths="/usr/share/fonts/truetype/msttcorefonts/"
)
for font in font_files:
    font_manager.fontManager.addfont(font)
    prop = font_manager.FontProperties(fname=font)

plt.rcParams["font.family"] = "Times New Roman"
mpl.rc("axes.formatter", use_mathtext=True)  # 1e-6 -> 10^6


def addlabels(x, y, fontsize=8, label_offset=0.0):
    for x_, y_ in zip(x, y):
        plt.text(x_, y_ + label_offset, y_, ha="center", fontsize=fontsize)


class MonetPlot:
    def __init__(
        self,
        path=None,
        xaxis=False,
        CI=False,
        label_axis="row",
        labels=None,
        x0=None,
        yy=None,
        CI_value=None,
    ):
        """
        Initializes the data handler with path and axis options for labels and confidence intervals.
        Allows for direct input of labels, x0, and yy if path is None.

        Parameters:
        path (str or None): The file path to the data file. If None, labels, x0, and yy must be provided as numpy arrays.
        xaxis (bool): If True, the first column (or first row) is used as x-axis values.
        CI (bool): If True, confidence interval (CI) values are used in the figures.
                   Ensure that the CI file is saved with the name <file_name>_CI.csv.
        label_axis (str): Specifies the axis where labels (classes) are located.
                          If set to "col", the first column is used for labels.
                          If set to "row", the first row is used for labels.
        labels (numpy array): Array of labels if path is None.
        x0 (numpy array): Array of x-axis values if path is None.
        yy (numpy array): 2D array of y-axis values if path is None.
        CI_value (numpy array or None): Array of confidence interval values if CI is True.
        """

        if path is None:
            # Check if labels, x0, and yy are provided directly
            if labels is None or x0 is None or yy is None:
                raise ValueError(
                    "When 'path' is None, 'labels', 'x0', and 'yy' must be provided as numpy arrays."
                )
            self.labels = labels
            self.x0 = x0
            self.yy = yy
            self.CI = CI_value if CI_value is not None else np.array([False])

        else:
            # Load data from the CSV file
            if label_axis == "col":
                data = self._csv_cleansing(path, "rows")
                labels = np.squeeze(data.iloc[:, :1].values, axis=1)
                x0 = data.columns.values[1:]
                yy = data.iloc[:, 1:].apply(pd.to_numeric).values
                if CI:
                    CI_path = path.split(".")[0] + "_CI.csv"
                    CI_data = pd.read_csv(CI_path, encoding="UTF8")
                    CI_value = CI_data.iloc[:, :].apply(pd.to_numeric).values.T
                    self.CI = CI_value
                else:
                    self.CI = np.array([False])
            else:
                data = self._csv_cleansing(path, "rows")
                if CI:
                    CI_path = path.split(".")[0] + "_CI.csv"
                    CI_data = pd.read_csv(CI_path, encoding="UTF8")
                    CI_value = CI_data.iloc[:, :].apply(pd.to_numeric).values.T
                    self.CI = CI_value
                else:
                    self.CI = np.array([False])
                if xaxis:
                    labels = data.columns.values[1:]
                    x0 = np.squeeze(data.iloc[:, :1].values, axis=1)
                    yy = data.iloc[:, 1:].apply(pd.to_numeric).values.T
                else:
                    labels = data.columns.values
                    x0 = np.array([1])  # just placeholder
                    yy = data.iloc[:, :].apply(pd.to_numeric).values.T
            self.labels = labels
            self.x0 = x0
            self.yy = yy

    def plot(
        self,
        marker=None,
        linestyle=None,
        grid=False,
        postfix="",
        ybins=None,
        xlabel=None,
        ylabel=None,
        legend=False,
        title=None,
        markersize=8,
        titlesize=18,
        skeleton=False,
        path=None,
        text_marker=False,
        shade=False,
        text_offset=(0.1, 0.2),
        text_size=10,
        bbox_anchor=(0.5, 1.1),
        ax=None,
    ):
        """
        Plots the stored data on a given matplotlib Axes object, or creates a new one if none is provided.

        This method supports multiple customization options, including error bars, shaded confidence intervals,
        inline text markers, grid display, and export to file. Designed for multi-series line plots with optional
        confidence intervals.

        Parameters
        ----------
        marker : str or None, optional
            Marker style for the plot. If set to `"enum"`, markers are automatically cycled across series.
        linestyle : str or None, optional
            Line style (e.g., '-', '--', ':', etc.). If None, default matplotlib style is used.
        grid : bool, default=False
            Whether to display horizontal grid lines on the y-axis.
        postfix : str, optional
            String to append to each numeric label when `text_marker` is enabled.
        ybins : int or None, optional
            Maximum number of y-axis ticks. Uses `MaxNLocator`.
        xlabel : str or None, optional
            Label for the x-axis.
        ylabel : str or None, optional
            Label for the y-axis.
        legend : bool, default=False
            Whether to display a legend above the plot.
        title : str or None, optional
            Title for the plot.
        markersize : int, default=8
            Size of the plot markers.
        titlesize : int, default=18
            Font size for the plot title.
        skeleton : bool, optional
            Reserved for future use (currently has no effect).
        path : str or None, optional
            If specified, the plot will be saved to this path as a PNG file.
        text_marker : bool, default=False
            Whether to annotate each point with its value as text.
        shade : bool, default=False
            If True and confidence intervals exist, shades the area between ±CI instead of showing error bars.
        text_offset : tuple of float, default=(0.1, 0.2)
            Offset applied to the x and y positions of the text annotations.
        text_size : int, default=10
            Font size of the text annotations.
        bbox_anchor : tuple of float, default=(0.5, 1.1)
            Anchor point for legend placement.
        ax : matplotlib.axes.Axes or None, optional
            Axes object to draw the plot on. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the plot.

        Notes
        -----
        - `self.yy` should be a list of y-value arrays for each series.
        - `self.x0` is a 1D array of x-values.
        - `self.CI` is a 2D array of confidence intervals (same shape as `yy`).
        - `self.labels` is a list of strings for legend labels.

        Examples
        --------
        >>> moneplot = MonetPlot(xaxis=True, x0=x, yy=y, labels=labels)
        >>> monetplot.plot(
        ...     marker="enum",
        ...     grid=True,
        ...     xlabel="Epoch",
        ...     ylabel="Accuracy",
        ...     legend=True,
        ...     title="Model Performance",
        ...     path="results/plot.png"
        ... )
        """

        CI_flag = True if self.CI.any() else False
        if marker == "enum":
            markers = itertools.cycle(("^", "D", "s", "o", "X"))

        with plt.style.context(genevive["plot"]):
            if ax is None:
                fig, ax = plt.subplots(figsize=(7.4, 4.8))
            for i, (y, label) in enumerate(zip(self.yy, self.labels)):
                m = next(markers) if marker == "enum" else marker

                if CI_flag and not shade:
                    ax.errorbar(
                        self.x0,
                        y,
                        yerr=self.CI[i],
                        marker=m,
                        markersize=markersize,
                        markeredgecolor="black",
                        capsize=3,
                        linestyle=linestyle,
                        linewidth=1,
                        label=label,
                    )
                else:
                    ax.plot(
                        self.x0,
                        y,
                        label=label,
                        marker=m,
                        linestyle=linestyle,
                        markersize=markersize,
                        markeredgecolor="black",
                    )
                    ax.tick_params(axis="x", labelsize=10)
                    if CI_flag:
                        ax.fill_between(
                            self.x0, y - self.CI[i], y + self.CI[i], alpha=0.2
                        )

                if text_marker:
                    line_color = ax.lines[-1].get_color()
                    for idx, (x_val, y_val) in enumerate(zip(self.x0, y)):
                        ax.text(
                            idx + float(text_offset[0]),
                            float(y_val) + float(text_offset[1]),
                            f"{y_val}" + postfix,
                            fontsize=text_size,
                            ha="center",
                            weight="bold",
                            color=line_color,
                        )

            if grid:
                ax.grid(axis="y")

            if ybins:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

            if xlabel:
                ax.set_xlabel(xlabel, weight="bold")
            if ylabel:
                ax.set_ylabel(ylabel, weight="bold")
            if title:
                ax.set_title(title, fontsize=titlesize, pad=35, weight="bold")
            if legend:
                ax.legend(
                    loc="upper center",
                    ncol=len(self.yy),
                    bbox_to_anchor=bbox_anchor,
                    frameon=False,
                )

            if path:
                plt.tight_layout()
                fig.savefig(path, dpi=300)

            return ax

    def two_yscale_plot(
        self,
        marker=None,
        linestyle=None,
        grid=False,
        xlabel=None,
        ylabel=None,
        title=None,
        markersize=7,
        titlesize=16,
        path=None,
        text_marker=False,
        text_offset=(0.1, 0.2),  # Tuple to control text positioning
        text_size=10,  # Font size for text markers
    ):
        """
        Plots two data series (x0, yy) on a single figure with two y-axes for distinct scaling.

        This method generates a line plot with two y-axes (`ax1` and `ax2`) on the same x-axis (`x0`) for comparing two
        data series with different y-axis scales. Each y-axis has an independent scale, making it ideal for contrasting
        data with differing units or magnitudes. Optional customizations include marker style, line style, grid display,
        axis labels, and title. If confidence interval data (`self.CI`) is available, shaded regions representing the
        intervals are added around each line.

        Parameters:
        ----------
        marker : str or None, default=None
            Specifies the marker style for the plot lines. If None, no marker is applied.

        linestyle : str or None, default=None
            Specifies the line style (e.g., "-", "--", "-.", ":"). If None, Matplotlib's default style is used.

        grid : bool, default=False
            If True, displays a grid on the plot background to improve readability.

        xtick_density : float, default=None
            Controls the density of major y-axis ticks. Values less than 1.0 decrease the number of x-ticks,
            while greater than 1.0 increase the number. For example, setting `xtick_ratio=0.5` halves the
            x-tick density, while `xtick_ratio=2.0 doubles it.

        xlabel : str or None, default=None
            Label for the x-axis. If None, no label is displayed.

        ylabel : str or None, default=None
            Not directly used, as `self.labels` is used to set the y-axis labels. If `self.labels` is unavailable,
            consider providing explicit y-axis labels here.

        title : str or None, default=None
            Title of the plot. If None, no title is displayed.

        markersize : int, default=7
            Size of the markers on the plot lines, if markers are applied.

        titlesize : int, default=12
            Font size for the title text.

        path : str or None, default=None
            If specified, saves the plot as an image to the provided file path with a resolution of 300 DPI.
            The format is inferred from the file extension.

        Notes:
        ------
        - This method uses a twin-axis (`ax1` and `ax2`) approach to plot two y-axes. Each axis is colored based on the
          data line color for easy identification.
        - Confidence intervals, if available in `self.CI`, are automatically shaded for each line plot.
        - The x-tick labels are set to display every other label, rotated vertically for readability.

        Example:
        --------
        CSV input :

        Indicator,2012,2013,2014,2015,...

        GDP,"1,440,111.40","1,500,819.10","1,562,928.90",...

        Growth rate,2.4,3.2,3.2,...

        >>> monet = MonetPlot("new_data/gdp.csv", xaxis=True, CI=False, label_axis='col')
        >>> monet.two_yscale_plot(grid=False, text_marker=True, xlabel="Year", marker="o", linestyle="-", title="",  markersize=4, path="figure2.png")
        """
        CI_flag = True if self.CI.any() else False
        with plt.style.context(genevive["plot"]):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax2._get_lines.get_next_color()
            ax1.spines["left"].set_visible(True)
            ax2.spines["right"].set_visible(True)
            plot1 = ax1.plot(
                self.x0,
                self.yy[0],
                label=self.labels[0],
                marker=marker,
                linestyle=linestyle,
                markersize=markersize,
                markeredgecolor="black",
            )
            if text_marker:
                line_color = plot1[0].get_color()
                scale_factor = 10**6
                for idx, (x_val, y_val) in enumerate(zip(self.x0, self.yy[0])):
                    ax1.text(
                        idx + float(text_offset[0]),  # Use index instead of x_val
                        float(y_val) + float(text_offset[1]) * (scale_factor / 10),
                        f"{float(y_val) /scale_factor:.2f}",
                        fontsize=text_size,
                        weight="bold",
                        ha="center",
                        color=line_color,
                    )

            plot2 = ax2.plot(
                self.x0,
                self.yy[1],
                label=self.labels[1],
                marker=marker,
                linestyle=linestyle,
                markersize=markersize,
                markeredgecolor="black",
            )

            if text_marker:
                line_color = plot2[0].get_color()
                for idx, (x_val, y_val) in enumerate(zip(self.x0, self.yy[1])):
                    ax2.text(
                        idx + float(text_offset[0]),  # Use index instead of x_val
                        float(y_val) + float(text_offset[1]),
                        f"{float(y_val):.2f}",
                        fontsize=text_size,
                        weight="bold",
                        ha="center",
                        color=line_color,
                    )

            if CI_flag:
                ax1.fill_between(
                    self.x0, self.yy[0] - self.CI[0], self.yy[0] + self.CI[0], alpha=0.2
                )
                ax2.fill_between(
                    self.x0, self.yy[1] - self.CI[1], self.yy[1] + self.CI[1], alpha=0.2
                )
            plt.xticks(self.x0[::2], rotation="vertical")

            if xlabel:
                ax1.set_xlabel(xlabel, weight="bold")

            ax1.set_ylabel(self.labels[0], color=plot1[0].get_color(), weight="bold")
            ax2.set_ylabel(self.labels[1], color=plot2[0].get_color(), weight="bold")
            if grid:
                plt.grid(axis="y")
            if title:
                plt.title(title, fontsize=titlesize, weight="bold")
            if path:
                plt.savefig(path, dpi=300)
            plt.plot()

    def histogram(
        self,
        density=False,
        xlabel=None,
        ylabel=None,
        title=None,
        titlesize=16,
        grid=False,
        bins=None,
        edge=None,
        path=None,
        figsize=None,
        ax=None,  # ✅ 추가됨
    ):
        """
        Plots one or more histograms on the given matplotlib Axes.

        This method generates histogram plots from the internal data (`self.yy`) and corresponding labels (`self.labels`),
        with options for customizing appearance, density normalization, bin edges, and export path.

        If no Axes object is provided, a new figure and axes will be created internally.

        Parameters
        ----------
        density : bool, default=False
            If True, the histogram will display a probability density (i.e., area sums to 1) instead of raw counts.
        xlabel : str or None, optional
            Label for the x-axis.
        ylabel : str or None, optional
            Label for the y-axis.
        title : str or None, optional
            Title for the histogram plot.
        titlesize : int, default=16
            Font size of the title.
        grid : bool, default=False
            Whether to show horizontal grid lines on the y-axis.
        bins : int or sequence of scalars or str, optional
            Number of bins or specific bin edges. If None, defaults to range(1994, 2025).
        edge : bool or None, optional
            If True, outlines each histogram bar with black edges.
        path : str or None, optional
            If provided, the figure will be saved to this path as a PNG file (dpi=300).
        figsize : tuple(int, int) or None, optional
            Size of the figure in inches (width, height). Ignored if `ax` is provided.
        ax : matplotlib.axes.Axes or None, optional
            Axes object to plot on. If None, a new Axes will be created.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the histogram(s).

        Notes
        -----
        - `self.yy` is assumed to be a list of 1D arrays (each array represents a sample distribution).
        - `self.labels` should contain a corresponding label for each dataset.
        - Each histogram is plotted overlaid in the same axes.
        - Bins should be chosen carefully to ensure visibility when multiple distributions are plotted.

        Examples
        --------
        >>> monetplot = MonetPlot("new_data/year.csv")
        >>> monetplot.histogram(density=False, ylabel="Count", grid=True, figsize=(19.2,4.8), title="Crime Case Distribution Over the Years", titlesize=20, path="APT/year2", edge=True)
        """

        if edge:
            edgecolor = "black"
        else:
            edgecolor = None

        with plt.style.context(genevive["plot"]):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)

            if bins is None:
                bins = range(1994, 2025, 1)

            for y, label in zip(self.yy, self.labels):
                ax.hist(
                    y,
                    bins=bins,
                    alpha=1.0,
                    density=density,
                    label=label,
                    edgecolor=edgecolor,
                    zorder=10,
                )

            if xlabel:
                ax.set_xlabel(xlabel, weight="bold")
            if ylabel:
                ax.set_ylabel(ylabel, weight="bold")
            if title:
                ax.set_title(title, fontsize=titlesize, weight="bold")
            if grid:
                ax.grid(axis="y", zorder=0)

            if path:
                plt.tight_layout()
                plt.savefig(path, dpi=300)

            return ax  # ✅ 외부에서 계속 쓸 수 있도록 리턴

    def multiple_bar(
        self,
        rotation=0,
        grid=False,
        show_value=False,
        postfix="",
        cgr=1.5,
        wr=1.0,
        xlabel=None,
        ylabel=None,
        title=None,
        minimal=False,
        titlesize=18,
        legendsize=10,
        barfontsize=10,
        legend=True,
        legend_cols=None,
        skeleton=False,
        path=None,
        ybins=None,
        bbox_to_anchor=None,
        hbar=False,
        inside_value=False,
        stacked=False,
        ylim=False,
        title_padding=15,
        ax=None,  # ✅ 추가됨
    ):
        """
        Plots a grouped or stacked bar chart (horizontal or vertical), with support for confidence intervals,
        data labels, layout control, and optional export.

        Supports both horizontal (`hbar=True`) and vertical bar layouts, with options for overlaying bars
        (grouped) or stacking them. Multiple series (from `self.yy`) are drawn with automatic positioning.
        This function is highly configurable and adaptable to a variety of bar visualization needs.

        Parameters
        ----------
        rotation : int, default=0
            Rotation angle of x-axis tick labels (used for vertical bar charts).
        grid : bool, default=False
            Whether to show axis grid lines.
        show_value : bool, default=False
            Whether to display the value as a text label on each bar.
        postfix : str, optional
            String to append to each bar label (e.g., '%' or 'pts').
        cgr : float, default=1.5
            Horizontal scaling factor (spacing between x-tick groups).
        wr : float, default=1.0
            Width ratio scaling for bar width and figure size.
        xlabel : str or None, optional
            Label for the x-axis.
        ylabel : str or None, optional
            Label for the y-axis.
        title : str or None, optional
            Plot title.
        minimal : bool, default=False
            If True, hides spines and y/x ticks for a cleaner look.
        titlesize : int, default=18
            Font size of the plot title.
        legendsize : int, default=10
            Font size of the legend labels.
        barfontsize : int, default=10
            Font size for value labels shown inside or outside bars.
        legend : bool, default=True
            Whether to show the legend.
        legend_cols : int or None, optional
            Number of columns for the legend. Defaults to the number of data series.
        skeleton : bool, optional
            Reserved for future use.
        path : str or None, optional
            If specified, saves the figure to the given path (PNG format, 300 dpi).
        ybins : int or None, optional
            Maximum number of y-axis ticks (using `MaxNLocator`).
        bbox_to_anchor : tuple(float, float) or None, optional
            Anchor position for legend placement.
        hbar : bool, default=False
            If True, plots horizontal bars (`barh`), else vertical bars.
        inside_value : bool, default=False
            If True, displays bar labels *inside* the bars; otherwise outside.
        stacked : bool, default=False
            If True, stacks bars instead of grouping them side-by-side.
        ylim : tuple or bool, optional
            Tuple (ymin, ymax) for y-axis limits. Only applies to vertical bars.
        title_padding : int, default=15
            Padding (in points) between the title and the top of the plot.
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw the chart on. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the bar plot.

        Notes
        -----
        - `self.yy` should be a list of arrays, each representing a series of bar heights.
        - `self.x0` contains the x-axis category labels.
        - `self.CI` (if defined) provides confidence intervals for each series.
        - Horizontal bars (`hbar=True`) use `ax.barh()` and reverse y-axis indexing for proper layout.

        Examples
        --------
        >>> fig, axs = plt.subplots(1, 2, figsize=(14,5))
        >>> moneplot = MonetPlot("new_data/5_3.csv", xaxis=True, label_axis='row', CI=False)
        >>> moneplot2 = MonetPlot("new_data/5_3_1.csv", xaxis=True, label_axis='row', CI=False)

        >>> moneplot.multiple_bar(grid=False, ax=axs[0], cgr=1.0, wr=1.2, ylabel="IAR (%)", show_value=True, minimal=False, legend=False, legend_cols=2, legendsize=12, path="rel_score.png", bbox_to_anchor=(1.0, -0.2), barfontsize=14, inside_value=True, title="IAR Shift in Sensitive Category")
        >>> moneplot2.multiple_bar(grid=False, ax=axs[1], cgr=1.0, wr=1.2, ylabel="IAR (%)", show_value=True, minimal=False, legend=False, legend_cols=1, legendsize=12, path="rel_score.png", bbox_to_anchor=(0.1, 1.10), barfontsize=14, inside_value=True, title="IAR Shift in Non-Sensitive Category")
        """

        CI_flag = True if self.CI.any() else False

        with plt.style.context(genevive["bar_vivid"]):
            if ax is None:
                fig, ax = plt.subplots(figsize=(6.4 * cgr * wr, 4.8))
            offsets = np.arange(len(self.x0)) * cgr * wr
            width = 1 / (1 + len(self.labels)) * wr

            if stacked:
                bottom = np.zeros(len(self.x0))
                for i, (y, label) in enumerate(zip(self.yy, self.labels)):
                    yerr = self.CI[i] if CI_flag else None
                    if hbar:
                        bar = ax.barh(
                            -offsets,
                            y,
                            width * 1.5,
                            left=bottom,
                            label=label,
                            xerr=yerr,
                            capsize=4,
                            zorder=2,
                        )
                        bottom += y
                    else:
                        bar = ax.bar(
                            offsets,
                            y,
                            width,
                            bottom=bottom,
                            label=label,
                            yerr=yerr,
                            capsize=4,
                            zorder=2,
                        )
                        bottom += y

                    if show_value:
                        ax.bar_label(
                            bar,
                            labels=[f"{val}" + postfix for val in y],
                            color="white",
                            fontsize=barfontsize,
                            weight="bold",
                            label_type="center",
                        )

            else:
                for i, (y, label) in enumerate(zip(self.yy, self.labels)):
                    yerr = self.CI[i] if CI_flag else None
                    if hbar:
                        bar = ax.barh(
                            -offsets - width * i,
                            y,
                            width,
                            xerr=yerr,
                            label=label,
                            capsize=4,
                            zorder=2,
                        )
                    else:
                        bar = ax.bar(
                            offsets + width * i,
                            y,
                            width,
                            yerr=yerr,
                            label=label,
                            capsize=4,
                            zorder=2,
                        )

                    if show_value:
                        text_labels = [
                            f"{val}" + postfix if val != 0 else "" for val in y
                        ]

                        if inside_value:
                            for rect, label_text in zip(bar, text_labels):
                                height = rect.get_height()
                                if label_text == "":
                                    continue
                                ax.text(
                                    rect.get_x() + rect.get_width() / 2,
                                    height
                                    - 2,  # ✅ 막대 내부 "끝쪽" (아래로 약간 내림)
                                    label_text,
                                    ha="center",
                                    va="top",  # 위 기준 정렬 (바깥쪽은 'bottom')
                                    color="white",
                                    fontsize=barfontsize,
                                    weight="bold",
                                )
                        else:
                            ax.bar_label(
                                bar,
                                labels=text_labels,
                                color="#333333",
                                fontsize=barfontsize,
                                weight="bold",
                                padding=1.5,
                            )

            centering = width * ((len(self.labels) - 1) / 2) if not stacked else 0

            if hbar:
                ax.set_yticks(-offsets - centering)
                ax.set_yticklabels(self.x0)
            else:
                ax.set_xticks(offsets + centering)
                ax.set_xticklabels(self.x0, rotation=rotation)

            if ybins:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

            if minimal:
                if hbar:
                    ax.spines["bottom"].set_visible(False)
                    ax.set_xticks([])
                else:
                    ax.spines["left"].set_visible(False)
                    ax.set_yticks([])
            else:
                ax.spines["left"].set_visible(False)
                ax.grid(axis="x" if hbar else "y", zorder=10)

            if grid:
                ax.grid(axis="x" if hbar else "y", zorder=10)

            if legend:
                ax.legend(
                    loc="upper center",
                    ncol=legend_cols if legend_cols else len(self.yy),
                    shadow=False,
                    bbox_to_anchor=bbox_to_anchor or (0.5, -0.1),
                    fontsize=legendsize,
                    frameon=False,
                )

            if hbar:
                if stacked:
                    gap = 0.5
                    offset = len(self.yy[0])
                    ax.set_ylim(-offset, gap)
            else:
                if ylim:
                    ax.set_ylim(ylim[0], ylim[1])

            if xlabel:
                ax.set_xlabel(xlabel, weight="bold")
            if ylabel:
                ax.set_ylabel(ylabel, weight="bold")
            if title:
                ax.set_title(
                    title, pad=title_padding, fontsize=titlesize, weight="bold"
                )

            if path:
                plt.tight_layout()
                plt.savefig(path, bbox_inches="tight", dpi=300)

            return ax

    def _adjust_xticks(self, plt, min_x, max_x, xtick_density):
        """Helper to set custom x-tick spacing."""
        x_ticks = plt.gca().get_xticks()
        if xtick_density >= 1.0:
            tick_interval = (x_ticks[1] - x_ticks[0]) / xtick_density
            new_x_ticks = np.arange(x_ticks[0], x_ticks[-1], tick_interval)
        else:
            num_xticks = len(x_ticks)
            after_num_xticks = int(np.ceil(num_xticks * xtick_density))
            interval = num_xticks // after_num_xticks
            new_x_ticks = x_ticks[::interval]
        plt.xticks(new_x_ticks)

    def _adjust_yticks(self, plt, min_y, max_y, ytick_density):
        """Helper to set custom y-tick spacing."""
        y_ticks = plt.gca().get_yticks()
        if ytick_density >= 1.0:
            tick_interval = (y_ticks[1] - y_ticks[0]) / ytick_density
            new_y_ticks = np.arange(y_ticks[0], y_ticks[-1], tick_interval)
        else:
            num_yticks = len(y_ticks)
            after_num_yticks = int(np.ceil(num_yticks * ytick_density))
            interval = num_yticks // after_num_yticks
            new_y_ticks = y_ticks[::interval]
        plt.yticks(new_y_ticks)
        plt.ylim(y_ticks[0], y_ticks[-1])

    def _csv_cleansing(self, path, axis):
        data = pd.read_csv(path, encoding="UTF8")
        data = data.replace("", np.nan)
        data = data.dropna(axis=axis, how="any")
        data[data.select_dtypes("object").columns] = data[
            data.select_dtypes("object").columns
        ].apply(lambda x: x.str.replace(",", ""))

        return data

    def stdout_data(self):
        print(f"Label: ({len(self.labels)})", self.labels)
        print(f"X0: {self.x0.shape}", self.x0)
        print(f"yy: {self.yy.shape}", self.yy[0])
        if self.CI.all():
            print(f"Confidence interval: {self.CI}")

    @staticmethod
    def display_array(array, cmap="Blues"):
        vmin, vmax = array.min(), array.max()
        fig, ax = plt.subplots()
        psm = ax.pcolormesh(array, cmap=cmap, rasterized=False, vmin=vmin, vmax=vmax)
        fig.colorbar(psm, ax=ax)
        plt.show()

    @staticmethod
    def array2csv(stack, tags, path):
        df = pd.DataFrame(stack, columns=tags)
        df.to_csv(path, index=False)

    @staticmethod
    def scatter(x, y, timestamp=None, colormap="RdPu_r", size=18, title=None):
        if type(timestamp) is np.ndarray:
            plt.scatter(x, y, c=timestamp, cmap=colormap, s=size)
            plt.colorbar()
        else:
            plt.scatter(x, y, s=size)
        if title:
            plt.title(title, fontsize=10)
        plt.show()
