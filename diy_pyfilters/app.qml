import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts

ApplicationWindow {
    visible: true
    title: "Virtual camera control center (VCCC)"
    property int margin: 11
    width: (mainLayout.implicitWidth + (2 * margin))
    height: (mainLayout.implicitHeight + (2 * margin))
    minimumWidth: mainLayout.Layout.implicitWidth + margin * 2
    minimumHeight: mainLayout.Layout.implicitHeight + margin * 2

    property QtObject enumerator
    property QtObject reader

    Connections {
        target: cameraSelector
        function onCurrentIndexChanged() {
            console.log("in cameraSelector");
            console.log("Selected camera: " + cameraSelector.model[cameraSelector.currentIndex])
            // enumerator.setUsedCamera(cameraSelector.model[cameraSelector.currentIndex])
            reader.set_camera(cameraSelector.model[cameraSelector.currentIndex])
        }
    }
    Connections {
        target: enumerator
        function onUpdated(cameras) {
            // console.log("Updated in qml " + cameras)
            
            cameraSelector.model = ["None"].concat(cameras);
            /*cameraSelector.model.clear()
            for (var i = 0; i < cameras.length; i++) {
                console.log("Adding camera " + cameras[i])
                cameraSelector.model.append({text: cameras[i]})
            }*/

            // cameraSelector.model = cameras
            /*cameraSelector.model.clear()
            for (var i = 0; i < cameras.length; i++) {
                let camera = cameras[i];
                cameraSelector.model.append({
                    id: camera[0],
                    displayText: camera[1] + ", " + camera[2][0] + "x" + camera[2][1],
                })
            }*/
        }
    }

    ColumnLayout {
        id: mainLayout
        anchors.fill: parent
        anchors.margins: margin
        GroupBox {
            id: rowBox
            title: "Camera"
            Layout.fillWidth: true

            RowLayout {
                id: rowLayout
                anchors.fill: parent
                Text { text: "Select camera:"; }
                ComboBox {
                    id: cameraSelector
                    Layout.fillWidth: true
                    model: ["None"]
                }
            }
        }
        GroupBox {
            title: "Effects"
            Layout.fillWidth: true

            RowLayout {
                // columns: 6
                // anchors.centerIn: parent
                // Layout.fillWidth: true
                // anchors.fill: parent

                Repeater {
                    model: ["None", "Sepia", "RedEye", "Star Wars", "Snowfall", "Mirror in the middle", "Hat"]
                    RadioButton {
                        text: modelData
                        checked: modelData === "None"
                        onClicked: {
                            if (checked) {
                                console.log("Clicked " + modelData)
                                reader.set_effect(modelData)
                            }
                        }
                    }
                }
            }
        }
        Item {
            // spacer
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
/*    GridLayout {
        columns: 2
        anchors.fill: parent

        Text { text: "Select camera:"; font.bold: true; }
        ComboBox {
            id: cameraSelector
            Layout.fillWidth: true
            model: ["Camera 1", "Camera 2", "Camera 3"]
        }
        Text { text: "words"; color: "red"; Layout.fillWidth: true }
        Text { text: "in"; font.underline: true }
        Text { text: "a"; font.pixelSize: 20 }
        Text { text: "row"; font.strikeout: true }
    }
    */
}