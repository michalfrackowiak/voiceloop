/* Static renderer for the deterministic P4-03 state embedded in index.html.
 * No network calls, no audio element, no external dependencies. All data is
 * SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN USABILITY RESULT.
 */
"use strict";

(function () {
  const states = JSON.parse(document.getElementById("review-state").textContent);
  const decision = JSON.parse(document.getElementById("go-no-go-state").textContent);
  const routingLabels = ["stt_wrong", "parser_wrong", "both_fine", "unusable_audio"];
  const allowedSpeeds = [0.5, 0.75, 1, 1.25, 1.5, 2];
  let index = 0;
  let player = null;
  let timer = null;
  let selectedRouting = "";
  let completed = false;
  let auditEvent = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function currentState() {
    return states[index];
  }

  function setText(id, text) {
    byId(id).textContent = text;
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function renderQueue() {
    const list = byId("item-list");
    clear(list);
    list.setAttribute("data-item-count", String(states.length));
    states.forEach(function (state, position) {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "item-button";
      button.dataset.itemId = state.item.item_id;
      button.setAttribute("aria-current", position === index ? "true" : "false");
      button.textContent = state.item.item_id;
      button.addEventListener("click", function () {
        selectItem(position);
      });
      li.appendChild(button);
      if (state.item.blind) {
        const badge = document.createElement("span");
        badge.className = "badge badge-blind";
        badge.textContent = "blind";
        li.appendChild(badge);
      }
      list.appendChild(li);
    });
  }

  function stopTimer() {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  function selectItem(position) {
    stopTimer();
    index = position;
    player = Object.assign({}, currentState().player, { bars: currentState().player.bars.slice() });
    selectedRouting = "";
    completed = false;
    auditEvent = null;
    byId("stage2-section").classList.toggle("blind-active", currentState().item.blind);
    renderQueue();
    renderPlayer();
    renderDiff();
    renderForm();
    renderBookingPanel();
    renderRouting();
    renderSecondOpinion();
    setText("corrected-transcript", "");
    byId("corrected-transcript").value = currentState().item.corrected_tokens.join(" ");
  }

  function renderPlayer() {
    const wave = byId("waveform");
    clear(wave);
    wave.setAttribute("data-bar-count", String(player.bars.length));
    wave.setAttribute("data-item-id", player.item_id);
    player.bars.forEach(function (bar) {
      const node = document.createElement("span");
      node.className = bar.low_confidence ? "wave-bar low-confidence" : "wave-bar";
      node.style.height = String(bar.amplitude) + "%";
      node.dataset.amplitude = String(bar.amplitude);
      node.dataset.lowConfidence = bar.low_confidence ? "true" : "false";
      node.title = bar.low_confidence ? "low_confidence_span" : "synthetic_bar";
      node.addEventListener("click", function () {
        seekTo(Math.floor((bar.index / player.bars.length) * player.duration_ms));
      });
      wave.appendChild(node);
    });
    currentState().player.spans.forEach(function (span) {
      wave.setAttribute("data-span-" + span.token_id, span.start_ms + "-" + span.end_ms);
    });
    updatePlayerReadout();
  }

  function updatePlayerReadout() {
    byId("position-readout").textContent = player.position_ms + " ms / " + player.duration_ms + " ms";
    byId("position-readout").dataset.positionMs = String(player.position_ms);
    byId("loop-count").textContent = "loops: " + player.loop_count;
    byId("loop-count").dataset.loopCount = String(player.loop_count);
    const toggle = byId("play-toggle");
    toggle.dataset.state = timer === null ? "paused" : "playing";
    toggle.textContent = timer === null ? "Play synthetic timeline" : "Pause synthetic timeline";
  }

  function seekTo(position) {
    player.position_ms = Math.max(0, Math.min(position, player.duration_ms));
    updatePlayerReadout();
  }

  function tick() {
    const advanced = Math.floor(100 * player.speed);
    player.position_ms += advanced;
    if (player.position_ms >= player.duration_ms) {
      if (player.loop) {
        player.position_ms = player.position_ms % player.duration_ms;
        player.loop_count += 1;
      } else {
        player.position_ms = player.duration_ms;
        stopTimer();
      }
    }
    updatePlayerReadout();
  }

  function renderDiff() {
    const view = byId("diff-view");
    clear(view);
    view.setAttribute("data-diff-ops", String(currentState().diff.length));
    currentState().diff.forEach(function (operation) {
      const token = document.createElement("span");
      token.className = "diff-token diff-" + operation.op;
      token.dataset.op = operation.op;
      token.textContent = operation.token;
      view.appendChild(token);
    });
  }

  function fieldValue(path, values) {
    const parts = path.split(".");
    let node = values;
    for (const part of parts) {
      if (node === null || typeof node !== "object") {
        return undefined;
      }
      node = node[part];
    }
    return node;
  }

  function validateField(field, value) {
    if (value === undefined || value === "" || value === null) {
      return field.required ? "required" : "";
    }
    if (field.json_type === "integer") {
      if (!/^-?\d+$/.test(String(value))) {
        return "expected_integer";
      }
      const number = Number(value);
      if (field.minimum !== null && number < field.minimum) {
        return "below_minimum";
      }
      if (field.maximum !== null && number > field.maximum) {
        return "above_maximum";
      }
      return "";
    }
    if (field.enum.length > 0 && field.enum.indexOf(String(value)) < 0) {
      return "not_in_enum";
    }
    if (field.min_length !== null && String(value).length < field.min_length) {
      return "too_short";
    }
    if (field.format === "date") {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(String(value))) {
        return "invalid_date";
      }
      const parsed = new Date(String(value) + "T00:00:00Z");
      if (Number.isNaN(parsed.getTime())) {
        return "invalid_date";
      }
    }
    if (field.format === "email" && !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(String(value))) {
      return "invalid_email";
    }
    return "";
  }

  function validateForm() {
    const descriptor = currentState().form;
    const values = collectValues();
    const issues = [];
    descriptor.fields.forEach(function (field) {
      const input = document.querySelector('[data-path="' + field.path + '"]');
      const code = validateField(field, fieldValue(field.path, values));
      if (input) {
        input.setAttribute("data-invalid", code ? "true" : "false");
      }
      if (code) {
        issues.push(field.path + ":" + code);
      }
    });
    const errors = document.querySelectorAll(".form-error");
    Array.prototype.forEach.call(errors, function (node) {
      node.textContent = "";
    });
    issues.forEach(function (issue) {
      const parts = issue.split(":");
      const input = document.querySelector('[data-path="' + parts[0] + '"]');
      if (input && input.parentNode) {
        const message = input.parentNode.querySelector(".form-error");
        if (message) {
          message.textContent = parts[1];
        }
      }
    });
    const start = fieldValue("check_in", values);
    const end = fieldValue("check_out", values);
    if (
      descriptor.cross_field_rules.indexOf("check_out_after_check_in") >= 0 &&
      typeof start === "string" &&
      typeof end === "string" &&
      end <= start
    ) {
      issues.push("check_out:check_out_after_check_in");
    }
    byId("slot-issues").setAttribute("data-issue-count", String(issues.length));
    setText("slot-issues", issues.length === 0 ? "" : issues.join(" "));
    completed = issues.length === 0 && selectedRouting !== "";
    renderSecondOpinion();
    return issues.length === 0;
  }

  function collectValues() {
    const values = {};
    currentState().form.fields.forEach(function (field) {
      const input = document.querySelector('[data-path="' + field.path + '"]');
      if (!input) {
        return;
      }
      const parts = field.path.split(".");
      let node = values;
      parts.slice(0, -1).forEach(function (part) {
        if (typeof node[part] !== "object" || node[part] === null) {
          node[part] = {};
        }
        node = node[part];
      });
      node[parts[parts.length - 1]] =
        field.json_type === "integer" ? (input.value === "" ? "" : Number(input.value)) : input.value;
    });
    return values;
  }

  function renderForm() {
    const form = byId("slot-form");
    clear(form);
    const descriptor = currentState().form;
    form.setAttribute("data-schema-version", descriptor.schema_version);
    descriptor.fields.forEach(function (field) {
      const wrapper = document.createElement("div");
      wrapper.className = "form-field";
      const label = document.createElement("label");
      label.textContent = field.path + (field.required ? " *" : "");
      label.htmlFor = "field-" + field.path.replace(/\./g, "-");
      const input = field.enum.length > 0 ? document.createElement("select") : document.createElement("input");
      input.id = label.htmlFor;
      input.type = field.json_type === "integer" ? "number" : "text";
      input.dataset.path = field.path;
      input.dataset.type = field.json_type;
      input.dataset.required = field.required ? "true" : "false";
      if (field.format) {
        input.dataset.format = field.format;
      }
      if (field.enum.length > 0) {
        field.enum.forEach(function (value) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = value;
          input.appendChild(option);
        });
      }
      if (!currentState().item.blind) {
        const value = fieldValue(field.path, currentState().item.prefill_slots);
        if (value !== undefined) {
          input.value = String(value);
        }
      }
      input.addEventListener("input", validateForm);
      const error = document.createElement("span");
      error.className = "form-error";
      error.dataset.path = field.path;
      wrapper.appendChild(label);
      wrapper.appendChild(input);
      wrapper.appendChild(error);
      form.appendChild(wrapper);
    });
    const submit = document.createElement("button");
    submit.type = "submit";
    submit.textContent = "Validate slots";
    form.appendChild(submit);
    form.onsubmit = function (event) {
      event.preventDefault();
      validateForm();
    };
    validateForm();
  }

  function renderBookingPanel() {
    const panel = byId("final-record");
    clear(panel);
    Object.keys(currentState().item.final_record)
      .sort()
      .forEach(function (field) {
        const term = document.createElement("dt");
        term.textContent = field;
        const value = document.createElement("dd");
        value.textContent = String(currentState().item.final_record[field]);
        panel.appendChild(term);
        panel.appendChild(value);
      });
    const list = byId("record-diff");
    clear(list);
    currentState().record_diff.forEach(function (row) {
      const li = document.createElement("li");
      li.className = row.status;
      li.dataset.field = row.field;
      li.dataset.status = row.status;
      li.textContent = row.field + ": " + row.status;
      list.appendChild(li);
    });
  }

  function renderRouting() {
    const container = byId("routing-buttons");
    clear(container);
    routingLabels.forEach(function (label) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.routing = label;
      button.setAttribute("aria-pressed", label === selectedRouting ? "true" : "false");
      button.textContent = label;
      button.addEventListener("click", function () {
        selectedRouting = label;
        renderRouting();
        validateForm();
      });
      container.appendChild(button);
    });
    byId("routing-selected").dataset.routing = selectedRouting;
    setText("routing-selected", selectedRouting);
  }

  function renderSecondOpinion() {
    const secondReviewer = byId("second-reviewer-id").value.trim();
    const approval = byId("approval-id").value.trim();
    const primary = currentState().item.reviewer_id;
    const reasons = [];
    if (!currentState().item.item_id) {
      reasons.push("missing_item_id");
    }
    if (!primary) {
      reasons.push("missing_primary_reviewer_id");
    }
    if (!secondReviewer) {
      reasons.push("missing_second_reviewer_id");
    }
    if (secondReviewer && secondReviewer === primary) {
      reasons.push("same_reviewer");
    }
    if (!approval) {
      reasons.push("missing_approval_id");
    } else if (!/^approval-token-\d+$/.test(approval)) {
      reasons.push("unrecognised_approval");
    }
    if (!completed) {
      reasons.push("review_not_completed");
    }
    const allowed = reasons.length === 0;
    const button = byId("second-opinion-request");
    button.disabled = !allowed;
    const output = byId("second-opinion-state");
    output.dataset.allowed = allowed ? "true" : "false";
    output.dataset.reasons = reasons.join(" ");
    setText("second-opinion-state", allowed ? "allowed" : "blocked");
  }

  function renderDecision() {
    const node = byId("go-no-go");
    node.textContent = JSON.stringify(decision, null, 2);
    node.dataset.decision = decision.decision;
  }

  byId("play-toggle").addEventListener("click", function () {
    if (timer !== null) {
      stopTimer();
    } else {
      timer = setInterval(tick, 100);
    }
    updatePlayerReadout();
  });
  byId("loop-toggle").addEventListener("click", function () {
    player.loop = !player.loop;
    const button = byId("loop-toggle");
    button.dataset.loop = player.loop ? "true" : "false";
    button.setAttribute("aria-pressed", player.loop ? "true" : "false");
    button.textContent = player.loop ? "Loop: on" : "Loop: off";
  });
  byId("speed-select").addEventListener("change", function (event) {
    const value = Number(event.target.value);
    if (allowedSpeeds.indexOf(value) < 0) {
      return;
    }
    player.speed = value;
    byId("speed-select").dataset.speed = String(value);
  });
  byId("second-reviewer-id").addEventListener("input", renderSecondOpinion);
  byId("approval-id").addEventListener("input", renderSecondOpinion);
  byId("second-opinion-request").addEventListener("click", function () {
    const output = byId("second-opinion-state");
    if (output.dataset.allowed !== "true") {
      return;
    }
    auditEvent = {
      status: "second_opinion_requested",
      item_id: currentState().item.item_id,
      label: "SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN USABILITY RESULT"
    };
    output.dataset.audit = JSON.stringify(auditEvent);
    setText("second-opinion-state", "second_opinion_requested");
  });

  renderDecision();
  selectItem(0);
})();
