import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

for (const file of process.argv.slice(2)) {
  const deck = await PresentationFile.importPptx(await FileBlob.load(file));
  const snapshot = await deck.inspect({
    kind: "slide,textbox,shape,image,table,chart,notes,layout",
    maxChars: 100000,
  });
  const safe = file.replace(/[^a-zA-Z0-9]/g, "_");
  await fs.writeFile(`.ppt_build/${safe}.ndjson`, snapshot.ndjson);
  console.log(file, deck.slides.items.length, deck.slideSize);
}
