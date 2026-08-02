Attribute VB_Name = "GraphicalAbstractEditable"
Option Explicit

Private Const SCALE_FACTOR As Double = 0.384
Private Const SLIDE_WIDTH As Single = 960
Private Const SLIDE_HEIGHT As Single = 384
Private Const SHAPE_PREFIX As String = "AITVBA_"

Sub BuildSkeleton()
    Dim pres As Presentation
    Dim sld As Slide
    Dim ids As Variant
    Dim boxes As Variant
    Dim index As Long

    Set pres = ActivePresentation
    PreparePresentation pres
    Set sld = PrepareSlide(pres)

    ids = Array("B01", "B02", "B03", "E01", "E02", "E03", "E04", "L01", "L02", "E05", "L03", "E06", "L04", "E07", "L05", "L06", "L07", "E08", "E09", "E10", "E11", "L08", "E12", "E13", "E14", "E15", "E16", "L09", "E17", "L10", "L11", "E18", "L12", "L13", "E19", "L14", "E20", "E21", "E22", "E23", "E24", "L15", "E25", "E26", "E27", "E28", "E29")
    boxes = Array( _
        Array(0, 0, 2500, 1000), Array(0, 0, 2500, 104), Array(0, 100, 2500, 4), Array(82, 23, 2336, 54), _
        Array(60, 148, 560, 776), Array(60, 148, 560, 78), Array(105, 246, 450, 40), Array(110, 309, 458, 122), _
        Array(112, 320, 454, 83), Array(126, 310, 230, 30), Array(140, 431, 400, 27), Array(94, 465, 492, 45), _
        Array(96, 527, 488, 1), Array(105, 543, 450, 40), Array(110, 608, 458, 153), Array(112, 624, 454, 106), _
        Array(112, 674, 454, 1), Array(126, 616, 260, 30), Array(315, 654, 216, 27), Array(94, 785, 492, 45), _
        Array(102, 847, 476, 52), Array(640, 532, 60, 8), Array(700, 148, 1160, 776), Array(700, 148, 1160, 78), _
        Array(790, 258, 970, 280), Array(830, 270, 890, 42), Array(815, 326, 190, 160), Array(1015, 402, 20, 8), _
        Array(1045, 326, 210, 160), Array(1080, 414, 140, 42), Array(1265, 402, 15, 8), Array(1290, 326, 220, 160), _
        Array(1320, 385, 162, 69), Array(1520, 402, 15, 8), Array(1545, 326, 190, 160), Array(1280, 496, 360, 88), _
        Array(808, 592, 944, 254), Array(840, 606, 880, 44), Array(850, 670, 402, 126), Array(1308, 670, 402, 126), _
        Array(820, 856, 920, 42), Array(1880, 532, 60, 8), Array(1940, 148, 500, 776), Array(1970, 164, 440, 56), _
        Array(1992, 296, 396, 224), Array(1992, 566, 396, 224), Array(1980, 832, 420, 46))

    For index = LBound(ids) To UBound(ids)
        AddSkeletonBox sld, CStr(ids(index)), CDbl(boxes(index)(0)), CDbl(boxes(index)(1)), CDbl(boxes(index)(2)), CDbl(boxes(index)(3))
    Next index
End Sub

Sub ReconstructFromImage()
    Dim pres As Presentation
    Dim sld As Slide

    Set pres = ActivePresentation
    PreparePresentation pres
    Set sld = PrepareSlide(pres)

    DrawCanvas sld
    DrawConventionalPanel sld
    DrawMainPanel sld
    DrawOutcomePanel sld
End Sub

Sub ExportCurrentSlidePng()
    Dim outputPath As String

    If Len(ActivePresentation.Path) = 0 Then
        Err.Raise vbObjectError + 1000, , "请先保存演示文稿，再导出 PNG。"
    End If

    outputPath = ActivePresentation.Path & "\Graphical_Abstract_ATE_editable_render.png"
    ActivePresentation.Slides(1).Export outputPath, "PNG", 2500, 1000
End Sub

Private Sub PreparePresentation(ByVal pres As Presentation)
    pres.PageSetup.SlideWidth = SLIDE_WIDTH
    pres.PageSetup.SlideHeight = SLIDE_HEIGHT
End Sub

Private Function PrepareSlide(ByVal pres As Presentation) As Slide
    Dim sld As Slide
    Dim index As Long

    If pres.Slides.Count = 0 Then
        Set sld = pres.Slides.Add(1, ppLayoutBlank)
    Else
        Set sld = pres.Slides(1)
    End If

    For index = sld.Shapes.Count To 1 Step -1
        If Left$(sld.Shapes(index).Name, Len(SHAPE_PREFIX)) = SHAPE_PREFIX Then
            sld.Shapes(index).Delete
        End If
    Next index

    Set PrepareSlide = sld
End Function

Private Sub DrawCanvas(ByVal sld As Slide)
    AddBox sld, "B01_Background", 0, 0, 2500, 1000, RGB(247, 250, 249), -1, 0, False, False
    AddBox sld, "B02_TitleBand", 0, 0, 2500, 104, RGB(255, 255, 255), -1, 0, False, False
    AddBox sld, "B03_TitleRule", 0, 100, 2500, 4, RGB(224, 232, 230), -1, 0, False, False
    AddText sld, "E01_Title", "Event-Triggered Predictive Reinforcement Learning with Unsupervised Dynamic Event Gating for Energy Efficient HVAC Control", 82, 66, 2336, 62, 14.2, RGB(23, 63, 70), True, ppAlignLeft, msoAnchorTop
End Sub

Private Sub DrawConventionalPanel(ByVal sld As Slide)
    Dim tickX As Double
    Dim index As Long

    AddBox sld, "E02_ContextPanel", 60, 148, 560, 776, RGB(255, 255, 255), RGB(213, 223, 220), 1.15, True, True, 18
    AddBox sld, "E03_ContextHeader", 60, 148, 560, 78, RGB(233, 240, 238), -1, 0, True, False, 18
    AddBox sld, "E03_ContextHeaderSquare", 60, 208, 560, 18, RGB(233, 240, 238), -1, 0, False, False
    AddText sld, "E03_ContextHeaderText", "Conventional trigger schemes", 75, 199, 530, 52, 13.05, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop

    AddText sld, "E04_FixedTitle", "Fixed-interval control", 105, 279, 450, 42, 9.98, RGB(38, 50, 56), True, ppAlignLeft, msoAnchorTop
    AddPlainLine sld, "L01_FixedAxisY", 110, 309, 110, 431, RGB(120, 144, 156), 1.15, False
    AddPlainLine sld, "L01_FixedAxisX", 110, 431, 568, 431, RGB(120, 144, 156), 1.15, False
    AddBezier sld, "L02_FixedLoad", Array(112, 403, 170, 398, 205, 397, 258, 390, 311, 383, 348, 327, 394, 339, 440, 351, 468, 378, 566, 320), RGB(0, 121, 107), 3.84, False
    AddText sld, "E05_CoolingLoad", "Cooling load", 126, 332, 230, 38, 8.45, RGB(69, 90, 100), False, ppAlignLeft, msoAnchorTop

    For index = 0 To 10
        tickX = 140 + index * 40
        AddPlainLine sld, "L03_Tick_" & Format$(index + 1, "00"), tickX, 431, tickX, 458, RGB(214, 79, 79), 1.92, False
    Next index

    AddBox sld, "E06_RedundantBand", 94, 465, 492, 45, RGB(247, 236, 234), -1, 0, True, False, 10
    AddText sld, "E06_RedundantText", "Redundant updates in stable periods", 100, 496, 480, 51, 10.37, RGB(167, 76, 72), True, ppAlignCenter, msoAnchorTop
    AddPlainLine sld, "L04_ContextDivider", 96, 527, 584, 527, RGB(213, 223, 220), 1.15, False

    AddText sld, "E07_StaticTitle", "Static-threshold control", 105, 576, 450, 44, 9.98, RGB(38, 50, 56), True, ppAlignLeft, msoAnchorTop
    AddPlainLine sld, "L05_StaticAxisY", 110, 608, 110, 761, RGB(120, 144, 156), 1.15, False
    AddPlainLine sld, "L05_StaticAxisX", 110, 761, 568, 761, RGB(120, 144, 156), 1.15, False
    AddBezier sld, "L06_StaticLoad", Array(112, 730, 170, 720, 210, 705, 258, 714, 306, 723, 340, 635, 392, 648, 444, 661, 472, 716, 566, 624), RGB(0, 121, 107), 3.84, False
    AddPlainLine sld, "L07_PresetThreshold", 112, 674, 566, 674, RGB(214, 79, 79), 2.69, True
    AddText sld, "E08_PresetThreshold", "Preset threshold", 126, 638, 260, 40, 8.45, RGB(69, 90, 100), False, ppAlignLeft, msoAnchorTop
    AddCircle sld, "E09_TriggerPoint1", 326, 670, 11, RGB(214, 79, 79), -1, 0
    AddCircle sld, "E09_TriggerPoint2", 520, 665, 11, RGB(214, 79, 79), -1, 0

    AddBox sld, "E10_DriftBand", 94, 785, 492, 45, RGB(247, 236, 234), -1, 0, True, False, 10
    AddText sld, "E10_DriftText", "Missed or excessive triggers under drift", 98, 816, 484, 51, 10.37, RGB(167, 76, 72), True, ppAlignCenter, msoAnchorTop
    AddBox sld, "E11_LimitedBand", 102, 847, 476, 52, RGB(255, 241, 214), RGB(233, 168, 37), 1.15, True, False, 12
    AddText sld, "E11_LimitedText", "Limited adaptability", 110, 882, 460, 62, 13.05, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop

    AddArrowLine sld, "L08_ToMain", 640, 536, 700, 536, RGB(38, 50, 56), 2.3
End Sub

Private Sub DrawMainPanel(ByVal sld As Slide)
    AddBox sld, "E12_MainPanel", 700, 148, 1160, 776, RGB(255, 255, 255), RGB(184, 207, 202), 1.15, True, True, 18
    AddBox sld, "E13_MainHeader", 700, 148, 1160, 78, RGB(220, 237, 234), -1, 0, True, False, 18
    AddBox sld, "E13_MainHeaderSquare", 700, 208, 1160, 18, RGB(220, 237, 234), -1, 0, False, False
    AddText sld, "E13_MainHeaderText", "Predict, detect, and act only when needed", 730, 199, 1100, 52, 13.05, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop

    AddBox sld, "E14_GateFrame", 790, 258, 970, 280, RGB(255, 250, 240), RGB(208, 139, 20), 1.54, True, False, 16
    AddText sld, "E15_GateTitle", "Unsupervised dynamic event gate", 830, 301, 890, 48, 10.75, RGB(0, 105, 92), True, ppAlignCenter, msoAnchorTop
    DrawInputCard sld
    DrawScoreCard sld
    DrawThresholdCard sld
    DrawDecisionCard sld

    AddElbowArrow sld, "L14_ToActions", Array(1640, 496, 1640, 558, 1280, 558, 1280, 584), RGB(38, 50, 56), 2.3
    AddBox sld, "E20_ActionFrame", 808, 592, 944, 254, RGB(248, 251, 250), RGB(184, 207, 202), 1.54, True, False, 18
    AddText sld, "E21_ActionTitle", "On-demand action selection", 840, 638, 880, 52, 9.98, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop

    AddBox sld, "E22_EventCard", 850, 670, 402, 126, RGB(0, 121, 107), -1, 0, True, False, 14
    AddText sld, "E22_EventLabel", "EVENT DETECTED", 870, 708, 362, 39, 8.45, RGB(255, 255, 255), False, ppAlignCenter, msoAnchorTop
    AddText sld, "E22_DqnLabel", "Predictive DQN", 870, 749, 362, 43, 9.98, RGB(255, 255, 255), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E22_UpdateLabel", "Update HVAC setpoint", 870, 780, 362, 35, 8.45, RGB(255, 255, 255), False, ppAlignCenter, msoAnchorTop

    AddBox sld, "E23_HoldCard", 1308, 670, 402, 126, RGB(233, 239, 237), RGB(120, 144, 156), 1.15, True, False, 14
    AddText sld, "E23_NoEventLabel", "NO EVENT", 1328, 708, 362, 39, 8.45, RGB(69, 90, 100), False, ppAlignCenter, msoAnchorTop
    AddText sld, "E23_HoldLabel", "Hold setpoint", 1328, 749, 362, 43, 9.98, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E23_ZeroOrderLabel", "Zero-order hold", 1328, 780, 362, 35, 8.45, RGB(69, 90, 100), False, ppAlignCenter, msoAnchorTop
    AddText sld, "E24_Responsive", "Responsive under disturbances " & ChrW(183) & " sparse during stable operation", 820, 888, 920, 55, 8.45, RGB(69, 90, 100), False, ppAlignCenter, msoAnchorTop

    AddArrowLine sld, "L15_ToOutcome", 1880, 536, 1940, 536, RGB(38, 50, 56), 2.3
End Sub

Private Sub DrawInputCard(ByVal sld As Slide)
    AddBox sld, "E16_InputCard", 815, 326, 190, 160, RGB(237, 246, 244), RGB(107, 159, 149), 1.15, True, False, 12
    AddText sld, "E16_InputStep", "INPUT", 830, 355, 160, 30, 6.53, RGB(0, 121, 107), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E16_InputState", "Predictive state s" & ChrW(8348), 825, 385, 170, 30, 7.68, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E16_InputLoad", "Load " & ChrW(183) & " weather", 825, 417, 170, 25, 6.91, RGB(69, 90, 100), False, ppAlignCenter, msoAnchorTop
    AddText sld, "E16_InputForecast", ChrW(183) & " forecast " & ChrW(183), 825, 440, 170, 25, 6.91, RGB(69, 90, 100), False, ppAlignCenter, msoAnchorTop
    AddBox sld, "E16_SharedStateBand", 846, 452, 128, 24, RGB(220, 237, 234), -1, 0, True, False
    AddText sld, "E16_SharedState", "shared state", 846, 470, 128, 30, 6.91, RGB(69, 90, 100), False, ppAlignCenter, msoAnchorTop
    AddArrowLine sld, "L09_InputToScore", 1015, 406, 1035, 406, RGB(38, 50, 56), 1.54
End Sub

Private Sub DrawScoreCard(ByVal sld As Slide)
    AddBox sld, "E17_ScoreCard", 1045, 326, 210, 160, RGB(255, 255, 255), RGB(123, 181, 170), 1.15, True, False, 12
    AddText sld, "E17_ScoreStep", "1 " & ChrW(183) & " SCORE", 1060, 355, 180, 30, 6.53, RGB(0, 121, 107), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E17_ScoreTitle", "Multi-scale scoring", 1055, 383, 190, 28, 7.68, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E17_S", "S", 1042, 419, 28, 24, 6.91, RGB(69, 90, 100), False, ppAlignRight, msoAnchorTop
    AddText sld, "E17_M", "M", 1042, 440, 28, 24, 6.91, RGB(69, 90, 100), False, ppAlignRight, msoAnchorTop
    AddText sld, "E17_L", "L", 1042, 461, 28, 24, 6.91, RGB(69, 90, 100), False, ppAlignRight, msoAnchorTop
    AddPlainLine sld, "L10_ShortScore", 1080, 414, 1155, 414, RGB(0, 150, 136), 2.69, False
    AddPlainLine sld, "L10_MediumScore", 1080, 435, 1190, 435, RGB(77, 182, 172), 2.69, False
    AddPlainLine sld, "L10_LongScore", 1080, 456, 1220, 456, RGB(128, 203, 196), 2.69, False
    AddText sld, "E17_FusedScore", "Fused score A(s" & ChrW(8348) & ")", 1055, 480, 190, 27, 6.91, RGB(69, 90, 100), False, ppAlignCenter, msoAnchorTop
    AddArrowLine sld, "L11_ScoreToThreshold", 1265, 406, 1280, 406, RGB(38, 50, 56), 1.54
End Sub

Private Sub DrawThresholdCard(ByVal sld As Slide)
    AddBox sld, "E18_ThresholdCard", 1290, 326, 220, 160, RGB(255, 255, 255), RGB(228, 180, 92), 1.15, True, False, 12
    AddText sld, "E18_AdaptStep", "2 " & ChrW(183) & " ADAPT", 1305, 355, 190, 30, 6.53, RGB(0, 121, 107), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E18_ThresholdTitle", "Adaptive thresholds", 1300, 383, 200, 28, 7.68, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop
    AddBezier sld, "L12_LocalThreshold", Array(1320, 438, 1352, 420, 1382, 431, 1410, 407, 1438, 383, 1460, 396, 1482, 385), RGB(214, 79, 79), 2.3, False
    AddBezier sld, "L12_GlobalThreshold", Array(1320, 450, 1362, 446, 1406, 442, 1482, 421), RGB(208, 139, 20), 2.3, True
    AddText sld, "E18_LocalLabel", "local", 1358, 414, 70, 28, 6.91, RGB(69, 90, 100), False, ppAlignLeft, msoAnchorTop
    AddText sld, "E18_GlobalLabel", "global", 1438, 454, 70, 28, 6.91, RGB(69, 90, 100), False, ppAlignLeft, msoAnchorTop
    AddText sld, "E18_DynamicThreshold", "Dynamic threshold " & ChrW(964) & ChrW(8348), 1300, 480, 200, 27, 6.91, RGB(69, 90, 100), False, ppAlignCenter, msoAnchorTop
    AddArrowLine sld, "L13_ThresholdToDecision", 1520, 406, 1535, 406, RGB(38, 50, 56), 1.54
End Sub

Private Sub DrawDecisionCard(ByVal sld As Slide)
    AddBox sld, "E19_DecisionCard", 1545, 326, 190, 160, RGB(255, 255, 255), RGB(123, 181, 170), 1.15, True, False, 12
    AddText sld, "E19_DecideStep", "3 " & ChrW(183) & " DECIDE", 1560, 355, 160, 30, 6.53, RGB(0, 121, 107), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E19_DecisionTitle", "Trigger decision", 1555, 383, 170, 28, 7.68, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E19_Formula", "A(s" & ChrW(8348) & ") > " & ChrW(964) & ChrW(8348), 1555, 418, 170, 31, 7.68, RGB(38, 50, 56), True, ppAlignCenter, msoAnchorTop
    AddBox sld, "E19_EventPill", 1564, 438, 83, 32, RGB(214, 79, 79), -1, 0, True, False, 16
    AddText sld, "E19_EventPillText", "EVENT", 1564, 460, 83, 36, 6.91, RGB(255, 255, 255), True, ppAlignCenter, msoAnchorTop
    AddBox sld, "E19_HoldPill", 1655, 438, 62, 32, RGB(233, 239, 237), RGB(120, 144, 156), 0.77, True, False, 16
    AddText sld, "E19_HoldPillText", "HOLD", 1655, 460, 62, 36, 6.53, RGB(0, 121, 107), True, ppAlignCenter, msoAnchorTop
End Sub

Private Sub DrawOutcomePanel(ByVal sld As Slide)
    AddBox sld, "E25_OutcomePanel", 1940, 148, 500, 776, RGB(23, 63, 70), -1, 0, True, True, 18
    AddText sld, "E26_OutcomeTitle", "Validated on real chiller data", 1970, 207, 440, 62, 13.05, RGB(255, 255, 255), True, ppAlignCenter, msoAnchorTop

    AddBox sld, "E27_UpdateMetricCard", 1992, 296, 396, 224, RGB(0, 121, 107), -1, 0, True, False, 14
    AddText sld, "E27_UpdateMetric", "53.54%", 2010, 393, 360, 78, 23.81, RGB(255, 255, 255), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E27_UpdateMetricLabel", "fewer action updates", 2010, 451, 360, 48, 8.83, RGB(255, 255, 255), True, ppAlignCenter, msoAnchorTop

    AddBox sld, "E28_EnergyMetricCard", 1992, 566, 396, 224, RGB(208, 139, 20), -1, 0, True, False, 14
    AddText sld, "E28_EnergyMetric", "2.13%", 2010, 663, 360, 78, 23.81, RGB(255, 255, 255), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E28_EnergyMetricLabel", "less energy use", 2010, 721, 360, 48, 8.83, RGB(255, 255, 255), True, ppAlignCenter, msoAnchorTop
    AddText sld, "E29_OutcomeFootnote", "Compared with fixed-step RL", 1980, 862, 420, 52, 8.45, RGB(255, 255, 255), False, ppAlignCenter, msoAnchorTop
End Sub

Private Function AddBox(ByVal sld As Slide, ByVal shapeId As String, ByVal x As Double, ByVal y As Double, ByVal width As Double, ByVal height As Double, ByVal fillColor As Long, ByVal lineColor As Long, ByVal lineWeight As Single, ByVal rounded As Boolean, ByVal hasShadow As Boolean, Optional ByVal cornerRadiusPx As Double = 12) As Shape
    Dim shp As Shape
    Dim shapeType As MsoAutoShapeType

    If rounded Then
        shapeType = msoShapeRoundedRectangle
    Else
        shapeType = msoShapeRectangle
    End If

    Set shp = sld.Shapes.AddShape(shapeType, Pt(x), Pt(y), Pt(width), Pt(height))
    shp.Name = SHAPE_PREFIX & shapeId
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = fillColor

    If lineColor < 0 Then
        shp.Line.Visible = msoFalse
    Else
        shp.Line.Visible = msoTrue
        shp.Line.ForeColor.RGB = lineColor
        shp.Line.Weight = lineWeight
    End If

    If rounded Then
        If width < height Then
            shp.Adjustments.Item(1) = cornerRadiusPx / width
        Else
            shp.Adjustments.Item(1) = cornerRadiusPx / height
        End If
    End If
    If hasShadow Then ApplySoftShadow shp
    Set AddBox = shp
End Function

Private Function AddText(ByVal sld As Slide, ByVal shapeId As String, ByVal value As String, ByVal x As Double, ByVal y As Double, ByVal width As Double, ByVal height As Double, ByVal fontSize As Single, ByVal fontColor As Long, ByVal isBold As Boolean, ByVal alignment As PpParagraphAlignment, ByVal verticalAnchor As MsoVerticalAnchor) As Shape
    Dim shp As Shape
    Dim fontSizePx As Double
    Dim topPx As Double
    Dim textHeightPx As Double

    fontSizePx = fontSize / SCALE_FACTOR
    topPx = y - 0.93 * fontSizePx
    textHeightPx = 1.25 * fontSizePx
    Set shp = sld.Shapes.AddTextbox(msoTextOrientationHorizontal, Pt(x), Pt(topPx), Pt(width), Pt(textHeightPx))
    shp.Name = SHAPE_PREFIX & shapeId
    shp.Fill.Visible = msoFalse
    shp.Line.Visible = msoFalse
    shp.TextFrame.MarginLeft = 0
    shp.TextFrame.MarginRight = 0
    shp.TextFrame.MarginTop = 0
    shp.TextFrame.MarginBottom = 0
    shp.TextFrame.WordWrap = msoFalse
    shp.TextFrame.AutoSize = ppAutoSizeNone
    shp.TextFrame.VerticalAnchor = verticalAnchor
    shp.TextFrame.TextRange.Text = value
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = alignment
    With shp.TextFrame.TextRange.Font
        .Name = "Arial"
        .Size = fontSize
        .Color.RGB = fontColor
        If isBold Then
            .Bold = msoTrue
        Else
            .Bold = msoFalse
        End If
    End With
    Set AddText = shp
End Function

Private Function AddPlainLine(ByVal sld As Slide, ByVal shapeId As String, ByVal x1 As Double, ByVal y1 As Double, ByVal x2 As Double, ByVal y2 As Double, ByVal lineColor As Long, ByVal lineWeight As Single, ByVal dashed As Boolean) As Shape
    Dim shp As Shape

    Set shp = sld.Shapes.AddLine(Pt(x1), Pt(y1), Pt(x2), Pt(y2))
    shp.Name = SHAPE_PREFIX & shapeId
    shp.Line.ForeColor.RGB = lineColor
    shp.Line.Weight = lineWeight
    If dashed Then shp.Line.DashStyle = msoLineDash
    Set AddPlainLine = shp
End Function

Private Function AddArrowLine(ByVal sld As Slide, ByVal shapeId As String, ByVal x1 As Double, ByVal y1 As Double, ByVal x2 As Double, ByVal y2 As Double, ByVal lineColor As Long, ByVal lineWeight As Single) As Shape
    Dim shp As Shape

    Set shp = AddPlainLine(sld, shapeId, x1, y1, x2, y2, lineColor, lineWeight, False)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    shp.Line.EndArrowheadLength = msoArrowheadShort
    shp.Line.EndArrowheadWidth = msoArrowheadWidthMedium
    Set AddArrowLine = shp
End Function

Private Function AddCircle(ByVal sld As Slide, ByVal shapeId As String, ByVal centerX As Double, ByVal centerY As Double, ByVal radius As Double, ByVal fillColor As Long, ByVal lineColor As Long, ByVal lineWeight As Single) As Shape
    Dim shp As Shape

    Set shp = sld.Shapes.AddShape(msoShapeOval, Pt(centerX - radius), Pt(centerY - radius), Pt(radius * 2), Pt(radius * 2))
    shp.Name = SHAPE_PREFIX & shapeId
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = fillColor
    If lineColor < 0 Then
        shp.Line.Visible = msoFalse
    Else
        shp.Line.ForeColor.RGB = lineColor
        shp.Line.Weight = lineWeight
    End If
    Set AddCircle = shp
End Function

Private Function AddBezier(ByVal sld As Slide, ByVal shapeId As String, ByVal points As Variant, ByVal lineColor As Long, ByVal lineWeight As Single, ByVal dashed As Boolean) As Shape
    Dim builder As FreeformBuilder
    Dim shp As Shape
    Dim index As Long

    Set builder = sld.Shapes.BuildFreeform(msoEditingCorner, Pt(CDbl(points(0))), Pt(CDbl(points(1))))
    index = 2
    Do While index <= UBound(points)
        If index + 5 <= UBound(points) Then
            builder.AddNodes msoSegmentCurve, msoEditingAuto, Pt(CDbl(points(index))), Pt(CDbl(points(index + 1))), Pt(CDbl(points(index + 2))), Pt(CDbl(points(index + 3))), Pt(CDbl(points(index + 4))), Pt(CDbl(points(index + 5)))
            index = index + 6
        ElseIf index + 1 <= UBound(points) Then
            builder.AddNodes msoSegmentLine, msoEditingAuto, Pt(CDbl(points(index))), Pt(CDbl(points(index + 1)))
            index = index + 2
        Else
            Exit Do
        End If
    Loop

    Set shp = builder.ConvertToShape
    shp.Name = SHAPE_PREFIX & shapeId
    shp.Fill.Visible = msoFalse
    shp.Line.ForeColor.RGB = lineColor
    shp.Line.Weight = lineWeight
    shp.Line.BeginArrowheadStyle = msoArrowheadNone
    shp.Line.EndArrowheadStyle = msoArrowheadNone
    If dashed Then shp.Line.DashStyle = msoLineDash
    Set AddBezier = shp
End Function

Private Function AddElbowArrow(ByVal sld As Slide, ByVal shapeId As String, ByVal points As Variant, ByVal lineColor As Long, ByVal lineWeight As Single) As Shape
    Dim builder As FreeformBuilder
    Dim shp As Shape
    Dim index As Long

    Set builder = sld.Shapes.BuildFreeform(msoEditingCorner, Pt(CDbl(points(0))), Pt(CDbl(points(1))))
    For index = 2 To UBound(points) Step 2
        builder.AddNodes msoSegmentLine, msoEditingCorner, Pt(CDbl(points(index))), Pt(CDbl(points(index + 1)))
    Next index

    Set shp = builder.ConvertToShape
    shp.Name = SHAPE_PREFIX & shapeId
    shp.Fill.Visible = msoFalse
    shp.Line.ForeColor.RGB = lineColor
    shp.Line.Weight = lineWeight
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    shp.Line.EndArrowheadLength = msoArrowheadShort
    shp.Line.EndArrowheadWidth = msoArrowheadWidthMedium
    Set AddElbowArrow = shp
End Function

Private Sub ApplySoftShadow(ByVal shp As Shape)
    With shp.Shadow
        .Visible = msoFalse
        .ForeColor.RGB = RGB(38, 50, 56)
        .Transparency = 0.86
        .OffsetX = 0
        .OffsetY = 3.07
        .Blur = 3.84
    End With
End Sub

Private Sub AddSkeletonBox(ByVal sld As Slide, ByVal shapeId As String, ByVal x As Double, ByVal y As Double, ByVal width As Double, ByVal height As Double)
    Dim safeWidth As Double
    Dim safeHeight As Double

    safeWidth = width
    safeHeight = height
    If safeWidth < 4 Then safeWidth = 4
    If safeHeight < 4 Then safeHeight = 4
    AddBox sld, "Skeleton_" & shapeId, x, y, safeWidth, safeHeight, RGB(225, 225, 225), RGB(110, 110, 110), 0.6, False, False
    AddText sld, "SkeletonLabel_" & shapeId, shapeId, x, y, safeWidth, safeHeight, 6, RGB(55, 55, 55), False, ppAlignCenter, msoAnchorMiddle
End Sub

Private Function Pt(ByVal pixelValue As Double) As Single
    Pt = CSng(pixelValue * SCALE_FACTOR)
End Function