// @category Export
import java.io.*;
import java.util.List;
import java.util.ArrayList;
import java.util.TreeSet;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.address.Address;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.RefType;
import ghidra.program.model.symbol.FlowType;
import ghidra.program.model.mem.MemoryBlock;



public class ExportIndirectFlowsCustom extends GhidraScript {

    private String funcName(Address a) {
        Function f = getFunctionContaining(a);
        return f == null ? "" : f.getName();
    }

    private String formatAddr(Address a) {
        long raw = a.getOffset();
        boolean isARM = currentProgram.getLanguage().getProcessor()
                        .toString().equalsIgnoreCase("ARM");
        if (isARM) raw = raw & ~1L;
        return String.format("%08x", raw & 0xFFFFFFFFL);
    }

    private List<String> collectTargets(Address from) {
        Reference[] refs = getReferencesFrom(from);
        TreeSet<String> targets = new TreeSet<>();
        for (Reference r : refs) {
            RefType t = r.getReferenceType();
            if (t == null || t.isFallthrough()) continue;
            Address to = r.getToAddress();
            if (to == null) continue;
            targets.add(formatAddr(to));
        }
        return new ArrayList<>(targets);
    }

    private static String jsonStr(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        String outDir;
        if (args != null && args.length >= 1) outDir = args[0];
        else outDir = askDirectory("Output directory", "Select").getAbsolutePath();

        String binaryName = currentProgram.getName();
        String arch = currentProgram.getLanguage().getProcessor().toString();
        String outPath = outDir + File.separator + binaryName + "_ghidra.json";

        Listing listing = currentProgram.getListing();
        InstructionIterator it = listing.getInstructions(true);

        ArrayList<String> entries = new ArrayList<>();
        while (it.hasNext()) {
            Instruction ins = it.next();
            FlowType ft = ins.getFlowType();

            if (ft == null || !ft.isComputed()) continue;
            if (!(ft.isJump() || ft.isCall())) continue;

            Address a = ins.getAddress();

            MemoryBlock block = currentProgram.getMemory().getBlock(a);
            if (block != null && block.getName().startsWith(".plt")) continue;
            String fn = funcName(a);
            String type = ft.isJump() ? "jump" : "call";
            List<String> targets = collectTargets(a);

            StringBuilder tb = new StringBuilder("[");
            for (int i = 0; i < targets.size(); i++) {
                if (i > 0) tb.append(", ");
                tb.append(jsonStr(targets.get(i)));
            }
            tb.append("]");

            StringBuilder entry = new StringBuilder();
            entry.append("    {\n");
            entry.append("      \"address\": ").append(jsonStr(formatAddr(a))).append(",\n");
            entry.append("      \"type\": ").append(jsonStr(type)).append(",\n");
            entry.append("      \"function\": ").append(jsonStr(fn)).append(",\n");
            entry.append("      \"targets\": ").append(tb.toString()).append("\n");
            entry.append("    }");
            entries.add(entry.toString());
        }

        try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(outPath)))) {
            w.println("{");
            w.println("  \"binary\": " + jsonStr(binaryName) + ",");
            w.println("  \"arch\": " + jsonStr(arch) + ",");
            w.println("  \"indirect_flows\": [");
            for (int i = 0; i < entries.size(); i++) {
                w.print(entries.get(i));
                if (i < entries.size() - 1) w.print(",");
                w.println();
            }
            w.println("  ]");
            w.println("}");
        }

        println("Wrote " + entries.size() + " indirect flows to " + outPath);
    }
}
