param(
    [string]$OutputPath = "$PSScriptRoot\Graphical_Abstract_EAB_editable_final.pptx",
    [string]$RenderPath = "$PSScriptRoot\Graphical_Abstract_EAB_editable_final_preview.png"
)

$ErrorActionPreference = "Stop"
$scale = 0.384
$prefix = "AITVBA_"
$msoTrue = -1
$msoFalse = 0
$msoShapeRectangle = 1
$msoShapeRoundedRectangle = 5
$msoShapeOval = 9
$msoTextOrientationHorizontal = 1
$msoAnchorMiddle = 3
$msoArrowheadTriangle = 2
$msoArrowheadShort = 1
$msoArrowheadWidthMedium = 2
$msoLineDash = 4
$ppAlignLeft = 1
$ppAlignCenter = 2
$ppAlignRight = 3
$ppLayoutBlank = 12
$ppSaveAsOpenXMLPresentation = 24

function Convert-ToPoint([double]$Value) {
    return [single]($Value * $scale)
}

function Add-Box {
    param(
        [string]$Id,
        [double]$X,
        [double]$Y,
        [double]$Width,
        [double]$Height,
        [int]$FillColor,
        [Nullable[int]]$LineColor = $null,
        [single]$LineWeight = 0,
        [double]$Radius = 12,
        [switch]$Rounded,
        [switch]$Shadow
    )

    $shapeType = if ($Rounded) { $msoShapeRoundedRectangle } else { $msoShapeRectangle }
    $shape = $slide.Shapes.AddShape($shapeType, (Convert-ToPoint $X), (Convert-ToPoint $Y), (Convert-ToPoint $Width), (Convert-ToPoint $Height))
    $shape.Name = "$prefix$Id"
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $FillColor
    if ($null -eq $LineColor) {
        $shape.Line.Visible = $msoFalse
    } else {
        $shape.Line.Visible = $msoTrue
        $shape.Line.ForeColor.RGB = $LineColor.Value
        $shape.Line.Weight = $LineWeight
    }
    if ($Rounded) {
        $shape.Adjustments.Item(1) = $Radius / [Math]::Min($Width, $Height)
    }
    if ($Shadow) {
        $shape.Shadow.Visible = $msoFalse
        $shape.Shadow.ForeColor.RGB = 0x383226
        $shape.Shadow.Transparency = 0.86
        $shape.Shadow.OffsetX = 0
        $shape.Shadow.OffsetY = 3.07
        $shape.Shadow.Blur = 3.84
    }
    return $shape
}

function Add-Text {
    param(
        [string]$Id,
        [string]$Text,
        [double]$X,
        [double]$Y,
        [double]$Width,
        [double]$Height,
        [single]$FontSize,
        [int]$FontColor,
        [switch]$Bold,
        [int]$Alignment = $ppAlignCenter
    )

    $fontSizePx = $FontSize / $scale
    $topPx = $Y - 0.93 * $fontSizePx
    $textHeightPx = 1.25 * $fontSizePx
    $shape = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, (Convert-ToPoint $X), (Convert-ToPoint $topPx), (Convert-ToPoint $Width), (Convert-ToPoint $textHeightPx))
    $shape.Name = "$prefix$Id"
    $shape.Fill.Visible = $msoFalse
    $shape.Line.Visible = $msoFalse
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = $msoFalse
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame.VerticalAnchor = 1
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
    $shape.TextFrame.TextRange.Font.Name = "Arial"
    $shape.TextFrame.TextRange.Font.Size = $FontSize
    $shape.TextFrame.TextRange.Font.Color.RGB = $FontColor
    $shape.TextFrame.TextRange.Font.Bold = if ($Bold) { $msoTrue } else { $msoFalse }
    return $shape
}

function Add-Line {
    param(
        [string]$Id,
        [double]$X1,
        [double]$Y1,
        [double]$X2,
        [double]$Y2,
        [int]$Color,
        [single]$Weight,
        [switch]$Dashed,
        [switch]$Arrow
    )

    $shape = $slide.Shapes.AddLine((Convert-ToPoint $X1), (Convert-ToPoint $Y1), (Convert-ToPoint $X2), (Convert-ToPoint $Y2))
    $shape.Name = "$prefix$Id"
    $shape.Line.ForeColor.RGB = $Color
    $shape.Line.Weight = $Weight
    if ($Dashed) {
        $shape.Line.DashStyle = $msoLineDash
    }
    if ($Arrow) {
        $shape.Line.EndArrowheadStyle = $msoArrowheadTriangle
        $shape.Line.EndArrowheadLength = $msoArrowheadShort
        $shape.Line.EndArrowheadWidth = $msoArrowheadWidthMedium
    }
    return $shape
}

function Add-Circle {
    param(
        [string]$Id,
        [double]$CenterX,
        [double]$CenterY,
        [double]$Radius,
        [int]$FillColor
    )

    $shape = $slide.Shapes.AddShape($msoShapeOval, (Convert-ToPoint ($CenterX - $Radius)), (Convert-ToPoint ($CenterY - $Radius)), (Convert-ToPoint ($Radius * 2)), (Convert-ToPoint ($Radius * 2)))
    $shape.Name = "$prefix$Id"
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $FillColor
    $shape.Line.Visible = $msoFalse
    return $shape
}

function Add-CubicCurve {
    param(
        [string]$Id,
        [double[]]$Points,
        [int]$Color,
        [single]$Weight,
        [switch]$Dashed
    )

    $startX = $Points[0]
    $startY = $Points[1]
    $segmentIndex = 2
    $lineIndex = 1
    while ($segmentIndex + 5 -lt $Points.Count) {
        $control1X = $Points[$segmentIndex]
        $control1Y = $Points[$segmentIndex + 1]
        $control2X = $Points[$segmentIndex + 2]
        $control2Y = $Points[$segmentIndex + 3]
        $endX = $Points[$segmentIndex + 4]
        $endY = $Points[$segmentIndex + 5]
        $previousX = $startX
        $previousY = $startY
        for ($step = 1; $step -le 12; $step++) {
            $time = $step / 12.0
            $inverse = 1.0 - $time
            $currentX = $inverse * $inverse * $inverse * $startX + 3 * $inverse * $inverse * $time * $control1X + 3 * $inverse * $time * $time * $control2X + $time * $time * $time * $endX
            $currentY = $inverse * $inverse * $inverse * $startY + 3 * $inverse * $inverse * $time * $control1Y + 3 * $inverse * $time * $time * $control2Y + $time * $time * $time * $endY
            Add-Line -Id ("{0}_{1:00}" -f $Id, $lineIndex) -X1 $previousX -Y1 $previousY -X2 $currentX -Y2 $currentY -Color $Color -Weight $Weight -Dashed:$Dashed | Out-Null
            $previousX = $currentX
            $previousY = $currentY
            $lineIndex++
        }
        $startX = $endX
        $startY = $endY
        $segmentIndex += 6
    }
}

function Add-ElbowArrow {
    param(
        [string]$Id,
        [double[]]$Points,
        [int]$Color,
        [single]$Weight
    )

    $segmentCount = ($Points.Count / 2) - 1
    for ($index = 0; $index -lt $segmentCount; $index++) {
        Add-Line -Id ("{0}_{1:00}" -f $Id, ($index + 1)) -X1 $Points[$index * 2] -Y1 $Points[$index * 2 + 1] -X2 $Points[$index * 2 + 2] -Y2 $Points[$index * 2 + 3] -Color $Color -Weight $Weight -Arrow:($index -eq $segmentCount - 1) | Out-Null
    }
}

function Convert-Rgb([int]$Red, [int]$Green, [int]$Blue) {
    return $Red + 256 * $Green + 65536 * $Blue
}

$white = Convert-Rgb 255 255 255
$canvas = Convert-Rgb 247 250 249
$titleColor = Convert-Rgb 23 63 70
$ink = Convert-Rgb 38 50 56
$body = Convert-Rgb 69 90 100
$axis = Convert-Rgb 120 144 156
$teal = Convert-Rgb 0 121 107
$tealMedium = Convert-Rgb 0 150 136
$tealLight = Convert-Rgb 77 182 172
$tealPale = Convert-Rgb 128 203 196
$red = Convert-Rgb 214 79 79
$redText = Convert-Rgb 167 76 72
$amber = Convert-Rgb 208 139 20
$lightRule = Convert-Rgb 224 232 230
$panelLine = Convert-Rgb 213 223 220
$mainLine = Convert-Rgb 184 207 202

$powerPoint = New-Object -ComObject PowerPoint.Application
$powerPoint.Visible = $msoTrue
$presentation = $powerPoint.Presentations.Add()
$presentation.PageSetup.SlideWidth = 960
$presentation.PageSetup.SlideHeight = 384
$slide = $presentation.Slides.Add(1, $ppLayoutBlank)

Add-Box -Id "B01_Background" -X 0 -Y 0 -Width 2500 -Height 1000 -FillColor $canvas | Out-Null
Add-Box -Id "B02_TitleBand" -X 0 -Y 0 -Width 2500 -Height 104 -FillColor $white | Out-Null
Add-Box -Id "B03_TitleRule" -X 0 -Y 100 -Width 2500 -Height 4 -FillColor $lightRule | Out-Null
Add-Text -Id "E01_Title" -Text "Event-Triggered Predictive Reinforcement Learning with Unsupervised Dynamic Event Gating for Energy Efficient HVAC Control" -X 82 -Y 66 -Width 2336 -Height 62 -FontSize 14.2 -FontColor $titleColor -Bold -Alignment $ppAlignLeft | Out-Null

Add-Box -Id "E02_ContextPanel" -X 60 -Y 148 -Width 560 -Height 776 -FillColor $white -LineColor $panelLine -LineWeight 1.15 -Rounded -Shadow | Out-Null
Add-Box -Id "E03_ContextHeader" -X 60 -Y 148 -Width 560 -Height 78 -FillColor (Convert-Rgb 233 240 238) -Rounded | Out-Null
Add-Box -Id "E03_ContextHeaderSquare" -X 60 -Y 208 -Width 560 -Height 18 -FillColor (Convert-Rgb 233 240 238) | Out-Null
Add-Text -Id "E03_ContextHeaderText" -Text "Conventional trigger schemes" -X 75 -Y 199 -Width 530 -Height 52 -FontSize 13.05 -FontColor $ink -Bold | Out-Null
Add-Text -Id "E04_FixedTitle" -Text "Fixed-interval control" -X 105 -Y 279 -Width 450 -Height 42 -FontSize 9.98 -FontColor $ink -Bold -Alignment $ppAlignLeft | Out-Null
Add-Line -Id "L01_FixedAxisY" -X1 110 -Y1 309 -X2 110 -Y2 431 -Color $axis -Weight 1.15 | Out-Null
Add-Line -Id "L01_FixedAxisX" -X1 110 -Y1 431 -X2 568 -Y2 431 -Color $axis -Weight 1.15 | Out-Null
Add-CubicCurve -Id "L02_FixedLoad" -Points @(112,403,170,398,205,397,258,390,311,383,348,327,394,339,440,351,468,378,566,320) -Color $teal -Weight 3.84
Add-Text -Id "E05_CoolingLoad" -Text "Cooling load" -X 126 -Y 332 -Width 230 -Height 38 -FontSize 8.45 -FontColor $body -Alignment $ppAlignLeft | Out-Null
for ($index = 0; $index -le 10; $index++) {
    $tickX = 140 + $index * 40
    Add-Line -Id ("L03_Tick_{0:00}" -f ($index + 1)) -X1 $tickX -Y1 431 -X2 $tickX -Y2 458 -Color $red -Weight 1.92 | Out-Null
}
Add-Box -Id "E06_RedundantBand" -X 94 -Y 465 -Width 492 -Height 45 -FillColor (Convert-Rgb 247 236 234) -Rounded | Out-Null
Add-Text -Id "E06_RedundantText" -Text "Redundant updates in stable periods" -X 100 -Y 496 -Width 480 -Height 51 -FontSize 10.37 -FontColor $redText -Bold | Out-Null
Add-Line -Id "L04_ContextDivider" -X1 96 -Y1 527 -X2 584 -Y2 527 -Color $panelLine -Weight 1.15 | Out-Null
Add-Text -Id "E07_StaticTitle" -Text "Static-threshold control" -X 105 -Y 576 -Width 450 -Height 44 -FontSize 9.98 -FontColor $ink -Bold -Alignment $ppAlignLeft | Out-Null
Add-Line -Id "L05_StaticAxisY" -X1 110 -Y1 608 -X2 110 -Y2 761 -Color $axis -Weight 1.15 | Out-Null
Add-Line -Id "L05_StaticAxisX" -X1 110 -Y1 761 -X2 568 -Y2 761 -Color $axis -Weight 1.15 | Out-Null
Add-CubicCurve -Id "L06_StaticLoad" -Points @(112,730,170,720,210,705,258,714,306,723,340,635,392,648,444,661,472,716,566,624) -Color $teal -Weight 3.84
Add-Line -Id "L07_PresetThreshold" -X1 112 -Y1 674 -X2 566 -Y2 674 -Color $red -Weight 2.69 -Dashed | Out-Null
Add-Text -Id "E08_PresetThreshold" -Text "Preset threshold" -X 126 -Y 638 -Width 260 -Height 40 -FontSize 8.45 -FontColor $body -Alignment $ppAlignLeft | Out-Null
Add-Circle -Id "E09_TriggerPoint1" -CenterX 326 -CenterY 670 -Radius 11 -FillColor $red | Out-Null
Add-Circle -Id "E09_TriggerPoint2" -CenterX 520 -CenterY 665 -Radius 11 -FillColor $red | Out-Null
Add-Box -Id "E10_DriftBand" -X 94 -Y 785 -Width 492 -Height 45 -FillColor (Convert-Rgb 247 236 234) -Rounded | Out-Null
Add-Text -Id "E10_DriftText" -Text "Missed or excessive triggers under drift" -X 98 -Y 816 -Width 484 -Height 51 -FontSize 10.37 -FontColor $redText -Bold | Out-Null
Add-Box -Id "E11_LimitedBand" -X 102 -Y 847 -Width 476 -Height 52 -FillColor (Convert-Rgb 255 241 214) -LineColor (Convert-Rgb 233 168 37) -LineWeight 1.15 -Rounded | Out-Null
Add-Text -Id "E11_LimitedText" -Text "Limited adaptability" -X 110 -Y 882 -Width 460 -Height 62 -FontSize 13.05 -FontColor $ink -Bold | Out-Null
Add-Line -Id "L08_ToMain" -X1 640 -Y1 536 -X2 700 -Y2 536 -Color $ink -Weight 2.3 -Arrow | Out-Null

Add-Box -Id "E12_MainPanel" -X 700 -Y 148 -Width 1160 -Height 776 -FillColor $white -LineColor $mainLine -LineWeight 1.15 -Rounded -Shadow | Out-Null
Add-Box -Id "E13_MainHeader" -X 700 -Y 148 -Width 1160 -Height 78 -FillColor (Convert-Rgb 220 237 234) -Rounded | Out-Null
Add-Box -Id "E13_MainHeaderSquare" -X 700 -Y 208 -Width 1160 -Height 18 -FillColor (Convert-Rgb 220 237 234) | Out-Null
Add-Text -Id "E13_MainHeaderText" -Text "Predict, detect, and act only when needed" -X 730 -Y 199 -Width 1100 -Height 52 -FontSize 13.05 -FontColor $ink -Bold | Out-Null
Add-Box -Id "E14_GateFrame" -X 790 -Y 258 -Width 970 -Height 280 -FillColor (Convert-Rgb 255 250 240) -LineColor $amber -LineWeight 1.54 -Rounded | Out-Null
Add-Text -Id "E15_GateTitle" -Text "Unsupervised dynamic event gate" -X 830 -Y 301 -Width 890 -Height 48 -FontSize 10.75 -FontColor (Convert-Rgb 0 105 92) -Bold | Out-Null

Add-Box -Id "E16_InputCard" -X 815 -Y 326 -Width 190 -Height 160 -FillColor (Convert-Rgb 237 246 244) -LineColor (Convert-Rgb 107 159 149) -LineWeight 1.15 -Rounded | Out-Null
Add-Text -Id "E16_InputStep" -Text "INPUT" -X 830 -Y 355 -Width 160 -Height 30 -FontSize 6.53 -FontColor $teal -Bold | Out-Null
Add-Text -Id "E16_InputState" -Text "Predictive state sₜ" -X 825 -Y 385 -Width 170 -Height 30 -FontSize 7.68 -FontColor $ink -Bold | Out-Null
Add-Text -Id "E16_InputLoad" -Text "Load · weather" -X 825 -Y 417 -Width 170 -Height 25 -FontSize 6.91 -FontColor $body | Out-Null
Add-Text -Id "E16_InputForecast" -Text "· forecast ·" -X 825 -Y 440 -Width 170 -Height 25 -FontSize 6.91 -FontColor $body | Out-Null
Add-Box -Id "E16_SharedStateBand" -X 846 -Y 452 -Width 128 -Height 24 -FillColor (Convert-Rgb 220 237 234) -Rounded | Out-Null
Add-Text -Id "E16_SharedState" -Text "shared state" -X 846 -Y 470 -Width 128 -Height 30 -FontSize 6.91 -FontColor $body | Out-Null
Add-Line -Id "L09_InputToScore" -X1 1015 -Y1 406 -X2 1035 -Y2 406 -Color $ink -Weight 1.54 -Arrow | Out-Null

Add-Box -Id "E17_ScoreCard" -X 1045 -Y 326 -Width 210 -Height 160 -FillColor $white -LineColor (Convert-Rgb 123 181 170) -LineWeight 1.15 -Rounded | Out-Null
Add-Text -Id "E17_ScoreStep" -Text "1 · SCORE" -X 1060 -Y 355 -Width 180 -Height 30 -FontSize 6.53 -FontColor $teal -Bold | Out-Null
Add-Text -Id "E17_ScoreTitle" -Text "Multi-scale scoring" -X 1055 -Y 383 -Width 190 -Height 28 -FontSize 7.68 -FontColor $ink -Bold | Out-Null
Add-Text -Id "E17_S" -Text "S" -X 1042 -Y 419 -Width 28 -Height 24 -FontSize 6.91 -FontColor $body -Alignment $ppAlignRight | Out-Null
Add-Text -Id "E17_M" -Text "M" -X 1042 -Y 440 -Width 28 -Height 24 -FontSize 6.91 -FontColor $body -Alignment $ppAlignRight | Out-Null
Add-Text -Id "E17_L" -Text "L" -X 1042 -Y 461 -Width 28 -Height 24 -FontSize 6.91 -FontColor $body -Alignment $ppAlignRight | Out-Null
Add-Line -Id "L10_ShortScore" -X1 1080 -Y1 414 -X2 1155 -Y2 414 -Color $tealMedium -Weight 2.69 | Out-Null
Add-Line -Id "L10_MediumScore" -X1 1080 -Y1 435 -X2 1190 -Y2 435 -Color $tealLight -Weight 2.69 | Out-Null
Add-Line -Id "L10_LongScore" -X1 1080 -Y1 456 -X2 1220 -Y2 456 -Color $tealPale -Weight 2.69 | Out-Null
Add-Text -Id "E17_FusedScore" -Text "Fused score A(sₜ)" -X 1055 -Y 480 -Width 190 -Height 27 -FontSize 6.91 -FontColor $body | Out-Null
Add-Line -Id "L11_ScoreToThreshold" -X1 1265 -Y1 406 -X2 1280 -Y2 406 -Color $ink -Weight 1.54 -Arrow | Out-Null

Add-Box -Id "E18_ThresholdCard" -X 1290 -Y 326 -Width 220 -Height 160 -FillColor $white -LineColor (Convert-Rgb 228 180 92) -LineWeight 1.15 -Rounded | Out-Null
Add-Text -Id "E18_AdaptStep" -Text "2 · ADAPT" -X 1305 -Y 355 -Width 190 -Height 30 -FontSize 6.53 -FontColor $teal -Bold | Out-Null
Add-Text -Id "E18_ThresholdTitle" -Text "Adaptive thresholds" -X 1300 -Y 383 -Width 200 -Height 28 -FontSize 7.68 -FontColor $ink -Bold | Out-Null
Add-CubicCurve -Id "L12_LocalThreshold" -Points @(1320,438,1352,420,1382,431,1410,407,1438,383,1460,396,1482,385) -Color $red -Weight 2.3
Add-CubicCurve -Id "L12_GlobalThreshold" -Points @(1320,450,1362,446,1406,442,1482,421) -Color $amber -Weight 2.3 -Dashed
Add-Text -Id "E18_LocalLabel" -Text "local" -X 1358 -Y 414 -Width 70 -Height 28 -FontSize 6.91 -FontColor $body -Alignment $ppAlignLeft | Out-Null
Add-Text -Id "E18_GlobalLabel" -Text "global" -X 1438 -Y 454 -Width 70 -Height 28 -FontSize 6.91 -FontColor $body -Alignment $ppAlignLeft | Out-Null
Add-Text -Id "E18_DynamicThreshold" -Text "Dynamic threshold τₜ" -X 1300 -Y 480 -Width 200 -Height 27 -FontSize 6.91 -FontColor $body | Out-Null
Add-Line -Id "L13_ThresholdToDecision" -X1 1520 -Y1 406 -X2 1535 -Y2 406 -Color $ink -Weight 1.54 -Arrow | Out-Null

Add-Box -Id "E19_DecisionCard" -X 1545 -Y 326 -Width 190 -Height 160 -FillColor $white -LineColor (Convert-Rgb 123 181 170) -LineWeight 1.15 -Rounded | Out-Null
Add-Text -Id "E19_DecideStep" -Text "3 · DECIDE" -X 1560 -Y 355 -Width 160 -Height 30 -FontSize 6.53 -FontColor $teal -Bold | Out-Null
Add-Text -Id "E19_DecisionTitle" -Text "Trigger decision" -X 1555 -Y 383 -Width 170 -Height 28 -FontSize 7.68 -FontColor $ink -Bold | Out-Null
Add-Text -Id "E19_Formula" -Text "A(sₜ) > τₜ" -X 1555 -Y 418 -Width 170 -Height 31 -FontSize 7.68 -FontColor $ink -Bold | Out-Null
Add-Box -Id "E19_EventPill" -X 1564 -Y 438 -Width 83 -Height 32 -FillColor $red -Rounded | Out-Null
Add-Text -Id "E19_EventPillText" -Text "EVENT" -X 1564 -Y 460 -Width 83 -Height 36 -FontSize 6.91 -FontColor $white -Bold | Out-Null
Add-Box -Id "E19_HoldPill" -X 1655 -Y 438 -Width 62 -Height 32 -FillColor (Convert-Rgb 233 239 237) -LineColor $axis -LineWeight 0.77 -Rounded | Out-Null
Add-Text -Id "E19_HoldPillText" -Text "HOLD" -X 1655 -Y 460 -Width 62 -Height 36 -FontSize 6.53 -FontColor $teal -Bold | Out-Null
Add-ElbowArrow -Id "L14_ToActions" -Points @(1640,496,1640,558,1280,558,1280,584) -Color $ink -Weight 2.3

Add-Box -Id "E20_ActionFrame" -X 808 -Y 592 -Width 944 -Height 254 -FillColor (Convert-Rgb 248 251 250) -LineColor $mainLine -LineWeight 1.54 -Rounded | Out-Null
Add-Text -Id "E21_ActionTitle" -Text "On-demand action selection" -X 840 -Y 638 -Width 880 -Height 52 -FontSize 9.98 -FontColor $ink -Bold | Out-Null
Add-Box -Id "E22_EventCard" -X 850 -Y 670 -Width 402 -Height 126 -FillColor $teal -Rounded | Out-Null
Add-Text -Id "E22_EventLabel" -Text "EVENT DETECTED" -X 870 -Y 708 -Width 362 -Height 39 -FontSize 8.45 -FontColor $white | Out-Null
Add-Text -Id "E22_DqnLabel" -Text "Predictive DQN" -X 870 -Y 749 -Width 362 -Height 43 -FontSize 9.98 -FontColor $white -Bold | Out-Null
Add-Text -Id "E22_UpdateLabel" -Text "Update HVAC setpoint" -X 870 -Y 780 -Width 362 -Height 35 -FontSize 8.45 -FontColor $white | Out-Null
Add-Box -Id "E23_HoldCard" -X 1308 -Y 670 -Width 402 -Height 126 -FillColor (Convert-Rgb 233 239 237) -LineColor $axis -LineWeight 1.15 -Rounded | Out-Null
Add-Text -Id "E23_NoEventLabel" -Text "NO EVENT" -X 1328 -Y 708 -Width 362 -Height 39 -FontSize 8.45 -FontColor $body | Out-Null
Add-Text -Id "E23_HoldLabel" -Text "Hold setpoint" -X 1328 -Y 749 -Width 362 -Height 43 -FontSize 9.98 -FontColor $ink -Bold | Out-Null
Add-Text -Id "E23_ZeroOrderLabel" -Text "Zero-order hold" -X 1328 -Y 780 -Width 362 -Height 35 -FontSize 8.45 -FontColor $body | Out-Null
Add-Text -Id "E24_Responsive" -Text "Responsive under disturbances · sparse during stable operation" -X 820 -Y 888 -Width 920 -Height 55 -FontSize 8.45 -FontColor $body | Out-Null
Add-Line -Id "L15_ToOutcome" -X1 1880 -Y1 536 -X2 1940 -Y2 536 -Color $ink -Weight 2.3 -Arrow | Out-Null

Add-Box -Id "E25_OutcomePanel" -X 1940 -Y 148 -Width 500 -Height 776 -FillColor $titleColor -Rounded -Shadow | Out-Null
Add-Text -Id "E26_OutcomeTitle" -Text "Validated on real chiller data" -X 1970 -Y 207 -Width 440 -Height 62 -FontSize 13.05 -FontColor $white -Bold | Out-Null
Add-Box -Id "E27_UpdateMetricCard" -X 1992 -Y 296 -Width 396 -Height 224 -FillColor $teal -Rounded | Out-Null
Add-Text -Id "E27_UpdateMetric" -Text "53.54%" -X 2010 -Y 393 -Width 360 -Height 78 -FontSize 23.81 -FontColor $white -Bold | Out-Null
Add-Text -Id "E27_UpdateMetricLabel" -Text "fewer action updates" -X 2010 -Y 451 -Width 360 -Height 48 -FontSize 8.83 -FontColor $white -Bold | Out-Null
Add-Box -Id "E28_EnergyMetricCard" -X 1992 -Y 566 -Width 396 -Height 224 -FillColor $amber -Rounded | Out-Null
Add-Text -Id "E28_EnergyMetric" -Text "2.13%" -X 2010 -Y 663 -Width 360 -Height 78 -FontSize 23.81 -FontColor $white -Bold | Out-Null
Add-Text -Id "E28_EnergyMetricLabel" -Text "less energy use" -X 2010 -Y 721 -Width 360 -Height 48 -FontSize 8.83 -FontColor $white -Bold | Out-Null
Add-Text -Id "E29_OutcomeFootnote" -Text "Compared with fixed-step RL" -X 1980 -Y 862 -Width 420 -Height 52 -FontSize 8.45 -FontColor $white | Out-Null

$resolvedOutput = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
$resolvedRender = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RenderPath)
$presentation.SaveAs($resolvedOutput, $ppSaveAsOpenXMLPresentation)
$slide.Export($resolvedRender, "PNG", 2500, 1000)

[pscustomobject]@{
    status = "created"
    deck = $resolvedOutput
    render = $resolvedRender
    shape_count = $slide.Shapes.Count
    powerpoint_visible = $true
} | ConvertTo-Json